PRAGMA foreign_keys = ON;

CREATE TABLE crop (
    crop_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    scientific_name TEXT,
    aliases TEXT,
    description TEXT
);

CREATE TABLE region (
    region_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    city TEXT NOT NULL,
    province TEXT NOT NULL
);

CREATE TABLE disease (
    disease_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    aliases TEXT,
    pathogen TEXT,
    category TEXT,
    description TEXT,
    preferred_conditions TEXT,
    primary_doc_id TEXT
);

CREATE TABLE symptom (
    symptom_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    aliases TEXT,
    description TEXT
);

CREATE TABLE pesticide (
    pesticide_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    active_ingredient TEXT NOT NULL,
    mode_of_action TEXT NOT NULL,
    safe_interval_days INTEGER NOT NULL CHECK (safe_interval_days >= 0),
    label_note TEXT NOT NULL,
    primary_doc_id TEXT NOT NULL,
    is_simulated INTEGER NOT NULL CHECK (is_simulated IN (0, 1))
);

CREATE TABLE document (
    doc_id TEXT PRIMARY KEY,
    doc_type TEXT NOT NULL,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    region_id TEXT REFERENCES region(region_id),
    published_date TEXT NOT NULL,
    source TEXT NOT NULL,
    is_simulated INTEGER NOT NULL CHECK (is_simulated IN (0, 1))
);

CREATE TABLE field_case (
    case_id TEXT PRIMARY KEY,
    crop_id TEXT NOT NULL REFERENCES crop(crop_id),
    region_id TEXT NOT NULL REFERENCES region(region_id),
    observed_date TEXT NOT NULL,
    disease_id TEXT REFERENCES disease(disease_id),
    temperature_c REAL,
    humidity_pct INTEGER,
    severity TEXT NOT NULL CHECK (severity IN ('low', 'medium', 'high')),
    evidence_doc_id TEXT NOT NULL REFERENCES document(doc_id),
    is_target_subset INTEGER NOT NULL CHECK (is_target_subset IN (0, 1)),
    narrative TEXT NOT NULL
);

CREATE TABLE case_symptom (
    case_id TEXT NOT NULL REFERENCES field_case(case_id),
    symptom_id TEXT NOT NULL REFERENCES symptom(symptom_id),
    is_query_target INTEGER NOT NULL CHECK (is_query_target IN (0, 1)),
    PRIMARY KEY (case_id, symptom_id)
);

CREATE TABLE disease_symptom (
    disease_id TEXT NOT NULL REFERENCES disease(disease_id),
    symptom_id TEXT NOT NULL REFERENCES symptom(symptom_id),
    is_primary INTEGER NOT NULL CHECK (is_primary IN (0, 1)),
    relation_strength REAL NOT NULL,
    source_doc_id TEXT NOT NULL REFERENCES document(doc_id),
    PRIMARY KEY (disease_id, symptom_id)
);

CREATE TABLE disease_pesticide (
    disease_id TEXT NOT NULL REFERENCES disease(disease_id),
    pesticide_id TEXT NOT NULL REFERENCES pesticide(pesticide_id),
    recommendation_priority INTEGER NOT NULL,
    dosage_text TEXT NOT NULL,
    safe_interval_days INTEGER NOT NULL,
    source_doc_id TEXT NOT NULL REFERENCES document(doc_id),
    PRIMARY KEY (disease_id, pesticide_id)
);

CREATE INDEX idx_case_region_date ON field_case(region_id, observed_date);
CREATE INDEX idx_case_disease ON field_case(disease_id);
CREATE INDEX idx_case_symptom_symptom ON case_symptom(symptom_id, case_id);
CREATE INDEX idx_disease_pesticide_interval ON disease_pesticide(disease_id, safe_interval_days);
