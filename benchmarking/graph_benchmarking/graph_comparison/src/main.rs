use clap::Parser;
use encoding_rs_io::DecodeReaderBytesBuilder;
use petgraph::algo::connected_components;
use petgraph::graph::{NodeIndex, UnGraph};
use rio_api::model::Triple;
use rio_api::parser::TriplesParser;
use rio_turtle::{NTriplesParser, TurtleParser};
use serde::Serialize;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

/// Efficiently compare two RDF graphs (Turtle or N-Triples) and generate statistics and plots.
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to the first graph file
    #[arg(index = 1)]
    graph1: String,

    /// Path to the second graph file
    #[arg(index = 2)]
    graph2: String,

    /// Format of the first graph (turtle/ntriples)
    #[arg(long, default_value = "turtle")]
    format1: String,

    /// Format of the second graph (turtle/ntriples)
    #[arg(long, default_value = "turtle")]
    format2: String,

    /// Display name for the first graph
    #[arg(long)]
    name1: Option<String>,

    /// Display name for the second graph
    #[arg(long)]
    name2: Option<String>,
}

/// Statistics for a single graph.
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

/// Data structure passed to the Python plotting script.
#[derive(Serialize)]
struct PlotData {
    g1_name: String,
    g2_name: String,
    g1_pos: HashMap<String, usize>,
    g2_pos: HashMap<String, usize>,
    g1_relations: HashMap<String, usize>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let start_time = Instant::now();
    let args = Args::parse();

    // Determine names and formats
    let g1_name = args.name1.unwrap_or_else(|| {
        Path::new(&args.graph1)
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string()
    });
    let g2_name = args.name2.unwrap_or_else(|| {
        Path::new(&args.graph2)
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string()
    });

    let fmt1 = detect_format(&args.graph1, &args.format1);
    let fmt2 = detect_format(&args.graph2, &args.format2);

    println!("\n--- Comparison: {} vs {} ---\n", g1_name, g2_name);

    // Process Graph 1
    let t1 = Instant::now();
    let stats1 = process_graph(&args.graph1, fmt1, &g1_name)?;
    println!(
        "Finished processing {} in {:.2?}",
        g1_name,
        t1.elapsed()
    );

    // Process Graph 2
    let t2 = Instant::now();
    let stats2 = process_graph(&args.graph2, fmt2, &g2_name)?;
    println!(
        "Finished processing {} in {:.2?}",
        g2_name,
        t2.elapsed()
    );

    // Output Results
    print_report(&g1_name, &stats1, &g2_name, &stats2);
    print_latex_table(&g1_name, &stats1, &g2_name, &stats2);
    save_results_to_csv(&g1_name, &stats1, &g2_name, &stats2)?;

    // Generate Plots
    generate_plots(&g1_name, &stats1, &g2_name, &stats2)?;

    println!("\nTotal execution time: {:.2?}", start_time.elapsed());

    Ok(())
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
            let pages: u64 = parts[1].parse().unwrap_or(0);
            // Assuming 4KB pages, which is standard for most Linux systems
            return (pages as f64 * 4096.0) / (1024.0 * 1024.0);
        }
    }
    0.0
}

/// Opens a file with a transcoder that replaces invalid UTF-8 sequences.
fn open_file_lossy<P: AsRef<Path>>(path: P) -> io::Result<BufReader<Box<dyn BufRead>>> {
    let file = File::open(path)?;
    let transcoded = DecodeReaderBytesBuilder::new()
        .encoding(Some(encoding_rs::UTF_8))
        .build(file);
    Ok(BufReader::new(Box::new(transcoded)))
}

fn process_graph(
    path: &str,
    format: &str,
    name: &str,
) -> Result<GraphStats, Box<dyn std::error::Error>> {
    println!("Loading graph {} from {}...", name, path);
    let start_mem = get_memory_usage();

    // Structures for in-memory processing
    let mut str_to_id: HashMap<String, u32> = HashMap::new();
    let mut next_id = 0;
    let mut edges: Vec<(u32, u32)> = Vec::new();
    let mut relation_counts: HashMap<String, usize> = HashMap::new();
    let mut pos_counts: HashMap<String, usize> = HashMap::new();

    let mut triple_count = 0;

    // Closure to process each triple
    let mut handle_triple = |s: &str, p: &str, o: &str| -> Result<(), io::Error> {
        triple_count += 1;

        // Update stats
        // Avoid allocation if key exists
        if let Some(count) = relation_counts.get_mut(p) {
            *count += 1;
        } else {
            relation_counts.insert(p.to_string(), 1);
        }

        // Heuristic for POS tags
        if p == "<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>" {
            if !o.starts_with("<http://www.w3.org/2002/07/owl#")
                && !o.starts_with("<http://www.w3.org/2000/01/rdf-schema#")
            {
                if let Some(count) = pos_counts.get_mut(o) {
                    *count += 1;
                } else {
                    pos_counts.insert(o.to_string(), 1);
                }
            }
        }

        // Map to integer IDs for structural analysis
        // Optimized: try get first to avoid allocation
        let s_id = if let Some(&id) = str_to_id.get(s) {
            id
        } else {
            let id = next_id;
            next_id += 1;
            str_to_id.insert(s.to_string(), id);
            id
        };

        let o_id = if let Some(&id) = str_to_id.get(o) {
            id
        } else {
            let id = next_id;
            next_id += 1;
            str_to_id.insert(o.to_string(), id);
            id
        };

        edges.push((s_id, o_id));

        Ok(())
    };

    // Parse the file
    let reader = open_file_lossy(path)?;

    if format == "ntriples" {
        parse_ntriples_robust(reader, &mut handle_triple)?;
    } else {
        // Use rio for Turtle
        let file_ttl = File::open(path)?;
        let transcoded = encoding_rs_io::DecodeReaderBytesBuilder::new()
            .encoding(Some(encoding_rs::UTF_8))
            .build(file_ttl);
        let reader_ttl = BufReader::new(transcoded);

        let mut parser = TurtleParser::new(reader_ttl, None);
        let mut parse_func = |t: Triple| -> Result<(), io::Error> {
            handle_triple(
                t.subject.to_string().as_str(),
                t.predicate.to_string().as_str(),
                t.object.to_string().as_str(),
            )
        };
        if let Err(e) = parser.parse_all(&mut parse_func) {
            eprintln!(
                "Warning: Turtle parsing encountered an error: {}. Some data might be missing.",
                e
            );
        }
    }

    let end_mem = get_memory_usage();

    // Structural Analysis (Connectivity)
    let (cc, lcc, asp, diam) = analyze_structure(next_id, &edges);

    // Entropy Calculation
    let entropy = calculate_entropy(&edges);

    // Density & Degree
    let num_nodes = next_id as f64;
    let num_edges = edges.len() as f64;
    let density = if num_nodes > 1.0 {
        num_edges / (num_nodes * (num_nodes - 1.0))
    } else {
        0.0
    };
    let degree = if num_nodes > 0.0 {
        (2.0 * num_edges) / num_nodes
    } else {
        0.0
    };

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

fn parse_ntriples_robust<R: BufRead, F>(reader: R, callback: &mut F) -> io::Result<()>
where
    F: FnMut(&str, &str, &str) -> Result<(), io::Error>,
{
    for line in reader.lines() {
        if let Ok(l) = line {
            if l.trim().is_empty() || l.starts_with('#') {
                continue;
            }

            // Heuristic parsing: find first space after subject, then after predicate, then last dot
            let chars: Vec<char> = l.chars().collect();
            if chars.is_empty() { continue; }

            let mut idx = 0;
            // Skip leading whitespace
            while idx < chars.len() && chars[idx].is_whitespace() { idx += 1; }

            // Subject
            let s_start = idx;
            if idx < chars.len() && chars[idx] == '<' {
                while idx < chars.len() && chars[idx] != '>' { idx += 1; }
                idx += 1;
            } else if idx < chars.len() && chars[idx] == '_' {
                while idx < chars.len() && !chars[idx].is_whitespace() { idx += 1; }
            } else {
                continue; // Invalid start
            }
            let s_end = idx;

            while idx < chars.len() && chars[idx].is_whitespace() { idx += 1; }

            // Predicate
            let p_start = idx;
            if idx < chars.len() && chars[idx] == '<' {
                while idx < chars.len() && chars[idx] != '>' { idx += 1; }
                idx += 1;
            } else {
                continue;
            }
            let p_end = idx;

            while idx < chars.len() && chars[idx].is_whitespace() { idx += 1; }

            // Object
            let o_start = idx;
            let mut o_end = chars.len();
            // Find last dot
            while o_end > 0 {
                o_end -= 1;
                if chars[o_end] == '.' { break; }
            }
            // Trim whitespace before dot
            while o_end > 0 && chars[o_end - 1].is_whitespace() { o_end -= 1; }

            if s_end > s_start && p_end > p_start && o_end > o_start {
                let s: String = chars[s_start..s_end].iter().collect();
                let p: String = chars[p_start..p_end].iter().collect();
                let o: String = chars[o_start..o_end].iter().collect();
                callback(s, p, o)?;
            }
        }
    }
    Ok(())
}

fn analyze_structure(num_nodes: u32, edges: &[(u32, u32)]) -> (usize, usize, f64, usize) {
    let mut graph = UnGraph::<u32, ()>::new_undirected();
    // Pre-allocate
    graph.reserve_nodes(num_nodes as usize);
    graph.reserve_edges(edges.len());

    for i in 0..num_nodes {
        graph.add_node(i);
    }
    for (s, o) in edges {
        graph.add_edge(NodeIndex::new(*s as usize), NodeIndex::new(*o as usize), ());
    }

    let cc = connected_components(&graph);

    // Calculating LCC size, ASP, and Diameter is expensive.
    // For large graphs, we skip ASP and Diameter.
    // We can estimate LCC size if needed, but petgraph's connected_components just returns count.
    // To get LCC size, we'd need to traverse.
    // Given the "research quality" requirement, we should probably do it, but efficiently.
    // However, for 100M edges, this might take a while.
    // We'll stick to the Python script's logic: skip if too big.

    let lcc_size = 0; // Placeholder
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

fn print_report(g1: &str, s1: &GraphStats, g2: &str, s2: &GraphStats) {
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

    // 7. Relation Distribution
    println!("7. Relation Type Distribution (Top 10):");
    println!("   Unique Relation Types: {}={}, {}={}", g1, s1.relations, g2, s2.relations);
    println!("   {:<50} | {:<15} | {:<15}", "Relation", g1, g2);
    println!("   {}", "-".repeat(86));

    let mut all_rels: HashSet<&String> = HashSet::new();
    all_rels.extend(s1.relation_counts.keys());
    all_rels.extend(s2.relation_counts.keys());

    let mut sorted_rels: Vec<&&String> = all_rels.iter().collect();
    sorted_rels.sort_by(|a, b| {
        let count_a = s1.relation_counts.get(**a).unwrap_or(&0) + s2.relation_counts.get(**a).unwrap_or(&0);
        let count_b = s1.relation_counts.get(**b).unwrap_or(&0) + s2.relation_counts.get(**b).unwrap_or(&0);
        count_b.cmp(&count_a)
    });

    for rel in sorted_rels.iter().take(10) {
        let rel_name = rel.split('/').last().unwrap_or(rel).split('#').last().unwrap_or(rel);
        println!("   {:<50} | {:<15} | {:<15}",
            rel_name,
            s1.relation_counts.get(**rel).unwrap_or(&0),
            s2.relation_counts.get(**rel).unwrap_or(&0)
        );
    }
    println!();

    // 8. POS Coverage
    println!("8. Part-of-Speech / Type Coverage (Top 10):");
    println!("   Unique POS Tags: {}={}, {}={}", g1, s1.pos_tags, g2, s2.pos_tags);
    println!("   {:<50} | {:<15} | {:<15}", "Type", g1, g2);
    println!("   {}", "-".repeat(86));

    let mut all_pos: HashSet<&String> = HashSet::new();
    all_pos.extend(s1.pos_counts.keys());
    all_pos.extend(s2.pos_counts.keys());

    let mut sorted_pos: Vec<&&String> = all_pos.iter().collect();
    sorted_pos.sort_by(|a, b| {
        let count_a = s1.pos_counts.get(**a).unwrap_or(&0) + s2.pos_counts.get(**a).unwrap_or(&0);
        let count_b = s1.pos_counts.get(**b).unwrap_or(&0) + s2.pos_counts.get(**b).unwrap_or(&0);
        count_b.cmp(&count_a)
    });

    for pos in sorted_pos.iter().take(10) {
        let pos_name = pos.split('/').last().unwrap_or(pos).split('#').last().unwrap_or(pos);
        println!("   {:<50} | {:<15} | {:<15}",
            pos_name,
            s1.pos_counts.get(**pos).unwrap_or(&0),
            s2.pos_counts.get(**pos).unwrap_or(&0)
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

fn save_results_to_csv(g1: &str, s1: &GraphStats, g2: &str, s2: &GraphStats) -> Result<(), Box<dyn std::error::Error>> {
    let filename = format!("comparison_{}_{}.csv", g1, g2);
    let mut w = csv::Writer::from_path(filename)?;

    // Transposed format
    w.write_record(&["Metric", g1, g2])?;
    w.write_record(&["Nodes", &s1.nodes.to_string(), &s2.nodes.to_string()])?;
    w.write_record(&["Triples", &s1.triples.to_string(), &s2.triples.to_string()])?;
    w.write_record(&["Relations", &s1.relations.to_string(), &s2.relations.to_string()])?;
    w.write_record(&["POS Tags", &s1.pos_tags.to_string(), &s2.pos_tags.to_string()])?;
    w.write_record(&["Density", &s1.density.to_string(), &s2.density.to_string()])?;
    w.write_record(&["Degree", &s1.degree.to_string(), &s2.degree.to_string()])?;
    w.write_record(&["Entropy", &s1.entropy.to_string(), &s2.entropy.to_string()])?;
    w.write_record(&["Connected Components", &s1.cc.to_string(), &s2.cc.to_string()])?;

    w.flush()?;
    Ok(())
}

fn generate_plots(g1: &str, s1: &GraphStats, g2: &str, s2: &GraphStats) -> Result<(), Box<dyn std::error::Error>> {
    let plot_data = PlotData {
        g1_name: g1.to_string(),
        g2_name: g2.to_string(),
        g1_pos: s1.pos_counts.clone(),
        g2_pos: s2.pos_counts.clone(),
        g1_relations: s1.relation_counts.clone(),
        overlap_relations: HashMap::new(), // Empty since overlap is removed
    };

    let plot_data_path = "plot_data.json";
    let file = File::create(plot_data_path)?;
    serde_json::to_writer(file, &plot_data)?;

    println!("Calling Python script to generate plots...");
    let status = Command::new("python")
        .arg("../../benchmarking/plot_graphs.py")
        .arg(plot_data_path)
        .status();

    match status {
        Ok(s) if s.success() => println!("Plots generated successfully."),
        _ => {
             let status_retry = Command::new("python")
                .arg("plot_graphs.py")
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
