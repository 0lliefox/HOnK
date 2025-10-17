# Commonsense Combiner (Ontology Builder)
Project for combining multiple commonsense knowledge bases into one, single ontology, to aid research in achieving better type resolution for work carried out in [LaSSI](https://github.com/LogDS/LaSSI).

## Setting Up
### Database (PostgreSQL)
PostgreSQL must be installed, along with a database and user:

#### Installation
*Linux (Ubuntu tested)*
```bash
sudo apt install postgresql -y
sudo -u postgres psql
```

*macOS*
```bash
brew install postgresql
psql postgres
```

#### Setup
```postgresql
create database ontology_db;
create user fox with encrypted password 'drowssap';
grant all privileges on database ontology_db to fox;
\c ontology_db fox
grant all on schema public to fox;
exit
```


### Getting Supporting Files
For this particular research, the following Knowledge Bases (KBs) were used, this does not mean other KBs could not be incorporated to enhance the final ontology further.

The script below can be used, which will download and extract the files in the correct format: 
```bash
./supporting_files/download_commonsense.sh
```
Files used in our experiments can be found at OSF.io here: https://osf.io/8mqs4/?view_only=282c38027c8043d5abd76a98001c31fa, or can be retrieved manually using the methods below:
#### ConceptNet
[ConceptNet](https://github.com/commonsense/conceptnet5/) was downloaded where the English Wiktionary dump from 2025-09-20 was used in experiments.

#### Wiktionary
Pre-expanded data from [Wiktextract](https://github.com/tatuylonen/wiktextract?tab=readme-ov-file#pre-extracted-data), which was used for current experiments was from 2025-08-23, and was then filtered to only contain English terms.

#### WordNet
The WordNet dump file was originally found [here](https://doi.org/10.5281/zenodo.3739540), as an input for ConceptNet, therefore used here to ensure the best clustering possible.

[//]: # (```bash)

[//]: # (curl "http://ldf.fi/wordnet/data?graph=http://ldf.fi/wordnet/wn31" --output supporting_files/wordnet.ttl)

[//]: # (```)