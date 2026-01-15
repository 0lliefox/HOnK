use clap::Parser;
use encoding_rs_io::DecodeReaderBytesBuilder;
use petgraph::algo::connected_components;
use petgraph::graph::{NodeIndex, UnGraph};
use rio_api::model::Triple;
use rio_api::parser::TriplesParser;
use rio_turtle::TurtleParser;
use serde::Serialize;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

/// Tool to compare two RDF graphs, calculating a variety of metrics related to their structure, content, and overlap.
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to the first graph file.
    #[arg(index = 1)]
    graph1: String,

    /// Path to the second graph file.
    #[arg(index = 2)]
    graph2: String,

    /// Format of the first graph (e.g., "turtle", "ntriples").
    #[arg(long, default_value = "turtle")]
    format1: String,

    /// Format of the second graph (e.g., "turtle", "ntriples").
    #[arg(long, default_value = "turtle")]
    format2: String,

    /// Custom display name for the first graph.
    #[arg(long)]
    name1: Option<String>,

    /// Custom display name for the second graph.
    #[arg(long)]
    name2: Option<String>,
}

/// Holds the calculated statistics for a single graph.
#[derive(Debug, Default, Serialize)]
struct GraphStats {
    nodes: usize,
    triples: usize,
    relations: usize,
    pos_tags: usize,
    density: f64,
    degree: f64,
    entropy: f64,
    cc: usize,
    lcc: usize,
    asp: f64,
    diam: usize,
    memory_mb: f64,
    #[serde(skip)]
    relation_counts: HashMap<String, usize>,
    #[serde(skip)]
    pos_counts: HashMap<String, usize>,
}

/// Holds statistics about the overlap between the two graphs.
#[derive(Debug, Default, Serialize)]
struct OverlapStats {
    node_intersection: usize,
    node_jaccard: f64,
    edge_intersection: usize,
    edge_jaccard: f64,
    #[serde(skip)]
    overlap_relations: HashMap<String, usize>,
    #[serde(skip)]
    unique_to_g1_sample: Vec<String>,
    #[serde(skip)]
    unique_to_g2_sample: Vec<String>,
}

/// Data structure passed to the Python plotting script via JSON.
#[derive(Serialize)]
struct PlotData {
    g1_name: String,
    g2_name: String,
    g1_pos: HashMap<String, usize>,
    g2_pos: HashMap<String, usize>,
    g1_relations: HashMap<String, usize>,
    overlap_relations: HashMap<String, usize>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let start_time = Instant::now();
    let args = Args::parse();

    let g1_name = args.name1.unwrap_or_else(|| get_filename(&args.graph1));
    let g2_name = args.name2.unwrap_or_else(|| get_filename(&args.graph2));

    let fmt1 = detect_format(&args.graph1, &args.format1);
    let fmt2 = detect_format(&args.graph2, &args.format2);

    println!("\n--- Comparison: {} vs {} ---\n", g1_name, g2_name);

    let temp_dir = tempfile::tempdir()?;
    let temp_path = temp_dir.path();

    let t1 = Instant::now();
    let stats1 = process_graph(&args.graph1, fmt1, &g1_name, temp_path, "g1")?;
    println!("Finished processing {} in {:.2?}", g1_name, t1.elapsed());

    let t2 = Instant::now();
    let stats2 = process_graph(&args.graph2, fmt2, &g2_name, temp_path, "g2")?;
    println!("Finished processing {} in {:.2?}", g2_name, t2.elapsed());

    let t_overlap = Instant::now();
    let overlap = compute_overlap(temp_path)?;
    println!("Computed overlap metrics in {:.2?}", t_overlap.elapsed());

    print_report(&g1_name, &stats1, &g2_name, &stats2, &overlap);
    print_latex_table(&g1_name, &stats1, &g2_name, &stats2);
    save_results_to_csv(&g1_name, &stats1, &g2_name, &stats2, &overlap)?;
    generate_plots(&g1_name, &stats1, &g2_name, &stats2, &overlap)?;

    println!("\nTotal execution time: {:.2?}", start_time.elapsed());
    Ok(())
}

fn get_filename(path_str: &str) -> String {
    Path::new(path_str)
        .file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .to_string()
}

fn detect_format<'a>(path: &str, format_arg: &'a str) -> &'a str {
    if format_arg == "turtle" && path.ends_with(".nt") {
        "ntriples"
    } else {
        format_arg
    }
}

fn get_memory_usage() -> f64 {
    if let Ok(usage) = std::fs::read_to_string("/proc/self/statm") {
        let parts: Vec<&str> = usage.split_whitespace().collect();
        if parts.len() > 1 {
            if let Ok(pages) = parts[1].parse::<u64>() {
                return (pages as f64 * 4096.0) / (1024.0 * 1024.0);
            }
        }
    }
    0.0
}

fn open_file_lossy<P: AsRef<Path>>(path: P) -> io::Result<Box<dyn BufRead>> {
    let file = File::open(path)?;
    let transcoded = DecodeReaderBytesBuilder::new()
        .encoding(Some(encoding_rs::UTF_8))
        .build(file);
    Ok(Box::new(BufReader::new(transcoded)))
}

fn process_graph(
    path: &str,
    format: &str,
    name: &str,
    temp_dir: &Path,
    key: &str,
) -> Result<GraphStats, Box<dyn std::error::Error>> {
    println!("Loading graph {} from {}...", name, path);
    let start_mem = get_memory_usage();

    let mut str_to_id: HashMap<String, u32> = HashMap::new();
    let mut next_id = 0;
    let mut edges: Vec<(u32, u32)> = Vec::new();
    let mut relation_counts: HashMap<String, usize> = HashMap::new();
    let mut pos_counts: HashMap<String, usize> = HashMap::new();

    let nodes_path = temp_dir.join(format!("{}_nodes.txt", key));
    let edges_path = temp_dir.join(format!("{}_edges.txt", key));
    let pairs_path = temp_dir.join(format!("{}_pairs.txt", key));
    let sop_path = temp_dir.join(format!("{}_sop.txt", key));

    let mut nodes_writer = BufWriter::new(File::create(&nodes_path)?);
    let mut edges_writer = BufWriter::new(File::create(&edges_path)?);
    let mut pairs_writer = BufWriter::new(File::create(&pairs_path)?);
    let mut sop_writer = BufWriter::new(File::create(&sop_path)?);

    let mut triple_count = 0;

    let mut handle_triple = |s: &str, p: &str, o: &str| -> Result<(), io::Error> {
        triple_count += 1;

        *relation_counts.entry(p.to_string()).or_insert(0) += 1;

        if p == "<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>" {
            if !o.starts_with("<http://www.w3.org/2002/07/owl#")
                && !o.starts_with("<http://www.w3.org/2000/01/rdf-schema#")
            {
                *pos_counts.entry(o.to_string()).or_insert(0) += 1;
            }
        }

        let s_id = *str_to_id.entry(s.to_string()).or_insert_with(|| {
            let id = next_id;
            next_id += 1;
            id
        });
        let o_id = *str_to_id.entry(o.to_string()).or_insert_with(|| {
            let id = next_id;
            next_id += 1;
            id
        });

        edges.push((s_id, o_id));

        writeln!(nodes_writer, "{}", s)?;
        writeln!(nodes_writer, "{}", o)?;
        writeln!(edges_writer, "{}\t{}\t{}", s, p, o)?;
        writeln!(pairs_writer, "{}\t{}", s, o)?;
        writeln!(sop_writer, "{}\t{}\t{}", s, o, p)?;

        Ok(())
    };

    let reader = open_file_lossy(path)?;
    if format == "ntriples" {
        parse_ntriples_robust(reader, &mut handle_triple)?;
    } else {
        let mut parser = TurtleParser::new(reader, None);
        if let Err(e) = parser.parse_all(&mut |t| {
            handle_triple(
                &t.subject.to_string(),
                &t.predicate.to_string(),
                &t.object.to_string(),
            )
        }) {
            eprintln!("Warning: Turtle parsing error: {}. Some data may be missing.", e);
        }
    }

    nodes_writer.flush()?;
    edges_writer.flush()?;
    pairs_writer.flush()?;
    sop_writer.flush()?;

    sort_file(&nodes_path, true)?;
    sort_file(&edges_path, false)?;
    sort_file(&pairs_path, true)?;
    sort_file(&sop_path, false)?;

    let end_mem = get_memory_usage();
    let (cc, lcc, asp, diam) = analyze_structure(next_id, &edges);
    let entropy = calculate_entropy(&edges);

    let num_nodes = next_id as f64;
    let num_edges = edges.len() as f64;
    let density = if num_nodes > 1.0 { num_edges / (num_nodes * (num_nodes - 1.0)) } else { 0.0 };
    let degree = if num_nodes > 0.0 { (2.0 * num_edges) / num_nodes } else { 0.0 };

    Ok(GraphStats {
        nodes: next_id as usize,
        triples: triple_count,
        relations: relation_counts.len(),
        pos_tags: pos_counts.len(),
        density,
        degree,
        entropy,
        cc,
        lcc,
        asp,
        diam,
        memory_mb: end_mem - start_mem,
        relation_counts,
        pos_counts,
    })
}

fn parse_ntriples_robust<R: BufRead, F>(mut reader: R, callback: &mut F) -> io::Result<()>
where
    F: FnMut(&str, &str, &str) -> Result<(), io::Error>,
{
    let mut line_buf = String::new();
    while reader.read_line(&mut line_buf)? > 0 {
        let l = line_buf.trim();
        if !l.is_empty() && !l.starts_with('#') {
            if let Some(s_end) = l.find(' ') {
                let s = &l[..s_end];
                if let Some(p_end) = l[s_end + 1..].find(' ') {
                    let p_end_abs = s_end + 1 + p_end;
                    let p = &l[s_end + 1..p_end_abs];
                    if let Some(o_end) = l.rfind(" .") {
                        let o = &l[p_end_abs + 1..o_end];
                        callback(s, p, o)?;
                    }
                }
            }
        }
        line_buf.clear();
    }
    Ok(())
}

fn analyze_structure(num_nodes: u32, edges: &[(u32, u32)]) -> (usize, usize, f64, usize) {
    let mut graph = UnGraph::<(), ()>::new_undirected();
    graph.extend_with_edges(edges.iter().map(|(s, o)| (*s, *o)));

    let cc = connected_components(&graph);
    let lcc_size = 0;
    let asp = 0.0;
    let diam = 0;

    (cc, lcc_size, asp, diam)
}

fn calculate_entropy(edges: &[(u32, u32)]) -> f64 {
    let mut degree_counts = HashMap::new();
    for (s, o) in edges {
        *degree_counts.entry(*s).or_insert(0) += 1;
        *degree_counts.entry(*o).or_insert(0) += 1;
    }
    let total_degrees: usize = degree_counts.values().sum();
    if total_degrees > 0 {
        degree_counts.values().map(|&count| {
            let p = count as f64 / total_degrees as f64;
            if p > 0.0 { -p * p.log2() } else { 0.0 }
        }).sum()
    } else {
        0.0
    }
}

fn sort_file(path: &Path, unique: bool) -> io::Result<()> {
    let status = Command::new("sort")
        .arg(if unique { "-u" } else { "-s" })
        .arg("-o")
        .arg(path)
        .arg(path)
        .env("LC_ALL", "C")
        .status()?;

    if !status.success() {
        return Err(io::Error::new(io::ErrorKind::Other, "Sort command failed"));
    }
    Ok(())
}

fn compute_overlap(temp_dir: &Path) -> Result<OverlapStats, Box<dyn std::error::Error>> {
    println!("Calculating overlap metrics...");

    let count_metrics = |f1: PathBuf, f2: PathBuf| -> io::Result<(usize, usize)> {
        let r1 = BufReader::new(File::open(f1)?);
        let r2 = BufReader::new(File::open(f2)?);
        let mut lines1 = r1.lines();
        let mut lines2 = r2.lines();

        let mut val1 = lines1.next();
        let mut val2 = lines2.next();

        let mut intersection = 0;
        let mut union = 0;

        while let (Some(Ok(s1)), Some(Ok(s2))) = (&val1, &val2) {
            if s1 == s2 {
                intersection += 1;
                union += 1;
                val1 = lines1.next();
                val2 = lines2.next();
            } else if s1 < s2 {
                union += 1;
                val1 = lines1.next();
            } else {
                union += 1;
                val2 = lines2.next();
            }
        }

        while let Some(Ok(_)) = val1 { union += 1; val1 = lines1.next(); }
        while let Some(Ok(_)) = val2 { union += 1; val2 = lines2.next(); }

        Ok((intersection, union))
    };

    let (node_int, node_union) = count_metrics(temp_dir.join("g1_nodes.txt"), temp_dir.join("g2_nodes.txt"))?;
    let (edge_int, edge_union) = count_metrics(temp_dir.join("g1_edges.txt"), temp_dir.join("g2_edges.txt"))?;

    let mut overlap_relations = HashMap::new();
    let mut unique_to_g1_sample = Vec::new();
    let mut unique_to_g2_sample = Vec::new();

    let f1_edges = File::open(temp_dir.join("g1_edges.txt"))?;
    let f2_edges = File::open(temp_dir.join("g2_edges.txt"))?;
    let mut lines1 = BufReader::new(f1_edges).lines();
    let mut lines2 = BufReader::new(f2_edges).lines();
    let mut v1 = lines1.next();
    let mut v2 = lines2.next();

    while let (Some(Ok(s1)), Some(Ok(s2))) = (&v1, &v2) {
        if s1 == s2 {
            v1 = lines1.next();
            v2 = lines2.next();
        } else if s1 < s2 {
            if unique_to_g1_sample.len() < 5 { unique_to_g1_sample.push(s1.clone()); }
            v1 = lines1.next();
        } else {
            if unique_to_g2_sample.len() < 5 { unique_to_g2_sample.push(s2.clone()); }
            v2 = lines2.next();
        }
    }
    unique_to_g1_sample.extend(lines1.flatten().take(5 - unique_to_g1_sample.len()));
    unique_to_g2_sample.extend(lines2.flatten().take(5 - unique_to_g2_sample.len()));

    let f1_sop = File::open(temp_dir.join("g1_sop.txt"))?;
    let f2_pairs = File::open(temp_dir.join("g2_pairs.txt"))?;
    let mut lines1 = BufReader::new(f1_sop).lines();
    let mut lines2 = BufReader::new(f2_pairs).lines();
    let mut val1 = lines1.next();
    let mut val2 = lines2.next();

    while let (Some(Ok(l1)), Some(Ok(l2))) = (&val1, &val2) {
        let parts1: Vec<&str> = l1.split('\t').collect();
        if parts1.len() < 3 { val1 = lines1.next(); continue; }

        let pair1 = format!("{}\t{}", parts1[0], parts1[1]);

        if &pair1 == l2 {
            *overlap_relations.entry(parts1[2].to_string()).or_insert(0) += 1;
            val1 = lines1.next();
        } else if &pair1 < l2 {
            val1 = lines1.next();
        } else {
            val2 = lines2.next();
        }
    }

    Ok(OverlapStats {
        node_intersection: node_int,
        node_jaccard: if node_union > 0 { node_int as f64 / node_union as f64 } else { 0.0 },
        edge_intersection: edge_int,
        edge_jaccard: if edge_union > 0 { edge_int as f64 / edge_union as f64 } else { 0.0 },
        overlap_relations,
        unique_to_g1_sample,
        unique_to_g2_sample,
    })
}

fn print_report(g1: &str, s1: &GraphStats, g2: &str, s2: &GraphStats, ov: &OverlapStats) {
    println!("1. Basic Statistics:");
    println!("   Nodes: {}={}, {}={}", g1, s1.nodes, g2, s2.nodes);
    println!("   Edges: {}={}, {}={}", g1, s1.triples, g2, s2.triples);
    println!();
    println!("2. Memory Usage (Approximate):");
    println!("   {}: {:.2} MB", g1, s1.memory_mb);
    println!("   {}: {:.2} MB", g2, s2.memory_mb);
    println!();
    println!("3. Graph Density & Degree:");
    println!("   Density: {}={:.12}, {}={:.12}", g1, s1.density, g2, s2.density);
    println!("   Avg Degree: {}={:.2}, {}={:.2}", g1, s1.degree, g2, s2.degree);
    println!();
    println!("4. Node Entropy:");
    println!("   Entropy: {}={:.4}, {}={:.4}", g1, s1.entropy, g2, s2.entropy);
    println!();
    println!("5. Graph Structure:");
    println!("   Connected Components: {}={}, {}={}", g1, s1.cc, g2, s2.cc);
    println!();
    println!("6. Overlap:");
    println!("   Node Intersection: {}", ov.node_intersection);
    println!("   Node Jaccard: {:.4}", ov.node_jaccard);
    println!("   Edge Intersection: {}", ov.edge_intersection);
    println!("   Edge Jaccard: {:.4}", ov.edge_jaccard);
    println!();

    println!("   Sample Triples Unique to {}:", g1);
    for t in &ov.unique_to_g1_sample {
        println!("     {}", t.replace("\t", " "));
    }
    println!();
    println!("   Sample Triples Unique to {}:", g2);
    for t in &ov.unique_to_g2_sample {
        println!("     {}", t.replace("\t", " "));
    }
    println!();

    println!("7. Relation Type Distribution (Top 10):");
    println!("   Unique Relation Types: {}={}, {}={}", g1, s1.relations, g2, s2.relations);
    println!("   {:<50} | {:<15} | {:<15}", "Relation", g1, g2);
    println!("   {}", "-".repeat(86));

    let mut all_rels: HashSet<&String> = HashSet::new();
    all_rels.extend(s1.relation_counts.keys());
    all_rels.extend(s2.relation_counts.keys());

    let mut sorted_rels: Vec<&String> = all_rels.into_iter().collect();
    sorted_rels.sort_by(|a, b| {
        let count_a = s1.relation_counts.get(*a).unwrap_or(&0) + s2.relation_counts.get(*a).unwrap_or(&0);
        let count_b = s1.relation_counts.get(*b).unwrap_or(&0) + s2.relation_counts.get(*b).unwrap_or(&0);
        count_b.cmp(&count_a)
    });

    for rel in sorted_rels.iter().take(10) {
        let rel_name = rel.split('/').last().unwrap_or(rel).split('#').last().unwrap_or(rel);
        println!("   {:<50} | {:<15} | {:<15}",
            rel_name,
            s1.relation_counts.get(*rel).unwrap_or(&0),
            s2.relation_counts.get(*rel).unwrap_or(&0)
        );
    }
    println!();

    println!("8. Part-of-Speech / Type Coverage (Top 10):");
    println!("   Unique POS Tags: {}={}, {}={}", g1, s1.pos_tags, g2, s2.pos_tags);
    println!("   {:<50} | {:<15} | {:<15}", "Type", g1, g2);
    println!("   {}", "-".repeat(86));

    let mut all_pos: HashSet<&String> = HashSet::new();
    all_pos.extend(s1.pos_counts.keys());
    all_pos.extend(s2.pos_counts.keys());

    let mut sorted_pos: Vec<&String> = all_pos.into_iter().collect();
    sorted_pos.sort_by(|a, b| {
        let count_a = s1.pos_counts.get(*a).unwrap_or(&0) + s2.pos_counts.get(*a).unwrap_or(&0);
        let count_b = s1.pos_counts.get(*b).unwrap_or(&0) + s2.pos_counts.get(*b).unwrap_or(&0);
        count_b.cmp(&count_a)
    });

    for pos in sorted_pos.iter().take(10) {
        let pos_name = pos.split('/').last().unwrap_or(pos).split('#').last().unwrap_or(pos);
        println!("   {:<50} | {:<15} | {:<15}",
            pos_name,
            s1.pos_counts.get(*pos).unwrap_or(&0),
            s2.pos_counts.get(*pos).unwrap_or(&0)
        );
    }
    println!();
}

fn print_latex_table(g1: &str, s1: &GraphStats, g2: &str, s2: &GraphStats) {
    println!("\n--- LaTeX Table Output ---\n");

    let fmt_num = |n: usize| -> String {
        let s = n.to_string();
        let mut result = String::new();
        for (i, c) in s.chars().rev().enumerate() {
            if i > 0 && i % 3 == 0 {
                result.push(',');
            }
            result.push(c);
        }
        result.chars().rev().collect()
    };

    let fmt_density = |n: f64| -> String {
        if n == 0.0 { return "0".to_string(); }
        let exponent = n.abs().log10().floor() as i32;
        let mantissa = n / 10f64.powi(exponent);
        format!("${:.2} \\times 10^{{{}}}$", mantissa, exponent)
    };

    println!("\\begin{{table*}}[h!]");
    println!("    \\centering");
    println!("    \\caption{{Comparison of Dataset Overlaps}}");
    println!("    \\label{{table:dataset-comparison}}");
    println!("    \\small");
    println!("    \\begin{{tabularx}}{{\\linewidth}}{{@{{}} X r r r r r r r @{{}}}}");
    println!("        \\toprule");
    println!("        \\textbf{{Dataset}} & \\textbf{{\\#Triples}} & \\textbf{{\\#Nodes}} & \\textbf{{\\#Relations}} & \\textbf{{\\#POS}} & \\textbf{{Density}} & \\textbf{{Degree}} & \\textbf{{Entropy}}  \\\\");
    println!("        \\midrule");
    println!("        {} & {} & {} & {} & {} & {} & {:.4} & {:.4} \\\\",
        g1.replace("_", "\\_"), fmt_num(s1.triples), fmt_num(s1.nodes), fmt_num(s1.relations), fmt_num(s1.pos_tags), fmt_density(s1.density), s1.degree, s1.entropy);
    println!("        \\addlinespace[0.5em]");
    println!("        {} & {} & {} & {} & {} & {} & {:.4} & {:.4} \\\\",
        g2.replace("_", "\\_"), fmt_num(s2.triples), fmt_num(s2.nodes), fmt_num(s2.relations), fmt_num(s2.pos_tags), fmt_density(s2.density), s2.degree, s2.entropy);
    println!("        \\bottomrule");
    println!("    \\end{{tabularx}}");
    println!("\\end{{table*}}");
}

fn save_results_to_csv(g1: &str, s1: &GraphStats, g2: &str, s2: &GraphStats, ov: &OverlapStats) -> Result<(), Box<dyn std::error::Error>> {
    let filename = format!("comparison_{}_{}.csv", g1.replace(" ", "_"), g2.replace(" ", "_"));
    let mut w = csv::Writer::from_path(filename)?;

    w.write_record(&["Metric", g1, g2])?;
    w.write_record(&["Nodes", &s1.nodes.to_string(), &s2.nodes.to_string()])?;
    w.write_record(&["Triples", &s1.triples.to_string(), &s2.triples.to_string()])?;
    w.write_record(&["Relations", &s1.relations.to_string(), &s2.relations.to_string()])?;
    w.write_record(&["POS Tags", &s1.pos_tags.to_string(), &s2.pos_tags.to_string()])?;
    w.write_record(&["Density", &s1.density.to_string(), &s2.density.to_string()])?;
    w.write_record(&["Degree", &s1.degree.to_string(), &s2.degree.to_string()])?;
    w.write_record(&["Entropy", &s1.entropy.to_string(), &s2.entropy.to_string()])?;
    w.write_record(&["Connected Components", &s1.cc.to_string(), &s2.cc.to_string()])?;

    w.write_record(&["", "", ""])?;
    w.write_record(&["Overlap Metric", "Value", ""])?;
    w.write_record(&["Node Intersection", &ov.node_intersection.to_string(), ""])?;
    w.write_record(&["Node Jaccard", &ov.node_jaccard.to_string(), ""])?;
    w.write_record(&["Edge Intersection", &ov.edge_intersection.to_string(), ""])?;
    w.write_record(&["Edge Jaccard", &ov.edge_jaccard.to_string(), ""])?;

    w.flush()?;
    Ok(())
}

fn generate_plots(g1: &str, s1: &GraphStats, g2: &str, s2: &GraphStats, ov: &OverlapStats) -> Result<(), Box<dyn std::error::Error>> {
    println!("Generating plots for '{}' and '{}'...", g1, g2);
    let plot_data = PlotData {
        g1_name: g1.to_string(),
        g2_name: g2.to_string(),
        g1_pos: s1.pos_counts.clone(),
        g2_pos: s2.pos_counts.clone(),
        g1_relations: s1.relation_counts.clone(),
        overlap_relations: ov.overlap_relations.clone(),
    };

    let plot_data_path = "plot_data.json";
    let file = File::create(plot_data_path)?;
    serde_json::to_writer(file, &plot_data)?;

    println!("Calling Python script to generate plots...");
    let status = Command::new("python")
        .arg("../plot_results.py")
        .arg("--comparison-data")
        .arg(plot_data_path)
        .status();

    match status {
        Ok(s) if s.success() => println!("Plots generated successfully."),
        _ => {
             let status_retry = Command::new("python")
                .arg("plot_results.py")
                .arg("--comparison-data")
                .arg(plot_data_path)
                .status();
             if let Ok(s) = status_retry {
                 if !s.success() { eprintln!("Failed to generate plots."); }
             } else {
                 eprintln!("Failed to execute python script. Ensure matplotlib is installed.");
             }
        }
    }
    Ok(())
}
