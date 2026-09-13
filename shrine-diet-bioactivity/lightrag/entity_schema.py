"""
Domain entity type definitions for the Unified Diet Knowledge Graph.

Maps SQLite tables → LightRAG entity nodes and relationship edges.
Used by ingest_unified.py to generate rich, searchable descriptions
for each entity before insertion via ainsert_custom_kg().
"""

from __future__ import annotations

import json
import re
import sqlite3
from typing import Any

# Safe Neo4j label pattern: alphanumeric + underscore, starts with letter/underscore
_SAFE_LABEL_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def safe_label(value: str) -> str:
    """Validate a string is safe for use as a Neo4j label (no injection)."""
    if not _SAFE_LABEL_RE.match(value):
        raise ValueError(f"Unsafe Neo4j label: {value!r}")
    return value


# ---------------------------------------------------------------------------
# Entity type definitions
# ---------------------------------------------------------------------------

ENTITY_TYPES = {
    # NOTE: every non-None query includes ``ORDER BY <pk>`` so ``LIMIT N``
    # subsampling is reproducible across rebuilds of the underlying
    # SQLite DB. See ``lightrag-thin-adapter-pivot-v2.plan.md`` §
    # "Subsample reproducibility".
    "Herb": {
        "source_table": "herbs",
        "id_field": "id",
        "name_field": "scientific_name",
        "query": "SELECT * FROM herbs ORDER BY id",
    },
    "Compound": {
        "source_table": "compounds",
        "id_field": "id",
        "name_field": "name",
        "query": "SELECT * FROM compounds ORDER BY id",
    },
    "Food": {
        "source_table": "compound_foods",
        "id_field": "food_name",
        "name_field": "food_name",
        # query is built dynamically in extract_entities to handle optional nutrition_100g column
        "query": None,
        "query_builder": "build_food_query",
    },
    "Target": {
        "source_table": "targets",
        "id_field": "id",
        "name_field": "name",
        "query": "SELECT id, name, uniprot_id, gene_symbol, druggability_status FROM targets ORDER BY id",
    },
    "Disease": {
        # Phase 3 — sources from the canonical disease registry. Replaces the
        # multi-table aggregation that used build_disease_query (which is kept
        # for backward-compat in QUERY_BUILDERS but no longer referenced).
        "source_table": "diseases_canonical",
        "id_field": "id",
        "name_field": "preferred_name",
        "query": (
            "SELECT id, preferred_name, mesh_id, umls_id, icd10cm_id, hpo_id, "
            "source_origin FROM diseases_canonical ORDER BY id"
        ),
    },
    "Symptom": {
        "source_table": "symptoms",
        "id_field": "id",
        "name_field": "name",
        "query": "SELECT id, name, symptom_type, description FROM symptoms ORDER BY id",
    },
    # -- Phase 1 drug-bioactive bridge (ChEMBL evidence) --
    "BioactivityEvidence": {
        "source_table": "bioactivity_evidence",
        "id_field": "id",
        # name_field is also the id — descriptions provide the human-readable text.
        "name_field": "id",
        "query": (
            "SELECT id, compound_id, chembl_compound_id, chembl_target_id, "
            "target_pref_name, target_type, target_organism, activity_type, "
            "relation, value, units, pchembl, assay_confidence, chembl_doc_id, "
            "publication_year FROM bioactivity_evidence ORDER BY id"
        ),
    },
    # -- Phase 4 KEGG pathway overlay --
    "Pathway": {
        "source_table": "kegg_pathways",
        "id_field": "id",
        "name_field": "name",
        "query": (
            "SELECT id, name, organism, category, source FROM kegg_pathways "
            "ORDER BY id"
        ),
    },
    # -- Tenant entity types (clinical practice layer) --
    # These have no SQLite source — ingested via tenant API (Phase 4).
    "Protocol": {
        "source_table": None,
        "id_field": "name",
        "name_field": "name",
        "query": None,
    },
    "Intervention": {
        "source_table": None,
        "id_field": "name",
        "name_field": "name",
        "query": None,
    },
    "Outcome": {
        "source_table": None,
        "id_field": "name",
        "name_field": "name",
        "query": None,
    },
    "Biomarker": {
        "source_table": None,
        "id_field": "name",
        "name_field": "name",
        "query": None,
    },
}


# ---------------------------------------------------------------------------
# Relationship type definitions
# ---------------------------------------------------------------------------

RELATIONSHIP_TYPES = {
    "CONTAINS_COMPOUND": {
        "source_table": "herb_compounds",
        "src_type": "Herb",
        "tgt_type": "Compound",
        "query": (
            "SELECT h.scientific_name as src_name, c.name as tgt_name, "
            "hc.plant_part, hc.concentration_low_ppm, hc.concentration_high_ppm "
            "FROM herb_compounds hc "
            "JOIN herbs h ON hc.herb_id = h.id "
            "JOIN compounds c ON hc.compound_id = c.id "
            "ORDER BY hc.herb_id, hc.compound_id"
        ),
    },
    "FOUND_IN_FOOD": {
        "source_table": "compound_foods",
        "src_type": "Compound",
        "tgt_type": "Food",
        "query": (
            "SELECT c.name as src_name, cf.food_name as tgt_name, "
            "cf.content_value, cf.content_unit, cf.food_part "
            "FROM compound_foods cf "
            "JOIN compounds c ON cf.compound_id = c.id "
            "ORDER BY cf.compound_id, cf.food_name"
        ),
    },
    "TARGETS_PROTEIN": {
        "source_table": "compound_targets",
        "src_type": "Compound",
        "tgt_type": "Target",
        "query": (
            "SELECT c.name as src_name, t.name as tgt_name, "
            "ct.activity_value, ct.activity_type "
            "FROM compound_targets ct "
            "JOIN compounds c ON ct.compound_id = c.id "
            "JOIN targets t ON ct.target_id = t.id "
            "ORDER BY ct.compound_id, ct.target_id"
        ),
    },
    "ASSOCIATED_WITH_DISEASE": {
        "source_table": "target_diseases",
        "src_type": "Target",
        "tgt_type": "Disease",
        "query": (
            "SELECT t.name as src_name, td.disease_name as tgt_name, "
            "td.evidence_layer as evidence "
            "FROM target_diseases td "
            "JOIN targets t ON td.target_id = t.id "
            "ORDER BY td.target_id, td.disease_name"
        ),
    },
    "TREATS_SYMPTOM": {
        "source_table": "herb_symptoms",
        "src_type": "Herb",
        "tgt_type": "Symptom",
        "query": (
            "SELECT h.scientific_name as src_name, s.name as tgt_name "
            "FROM herb_symptoms hs "
            "JOIN herbs h ON hs.herb_id = h.id "
            "JOIN symptoms s ON hs.symptom_id = s.id "
            "ORDER BY hs.herb_id, hs.symptom_id"
        ),
    },
    # -- Phase 1 drug-bioactive bridge (ChEMBL evidence) --
    "HAS_EVIDENCE": {
        "source_table": "bioactivity_evidence",
        "src_type": "Compound",
        "tgt_type": "BioactivityEvidence",
        "query": (
            "SELECT c.name AS src_name, be.id AS tgt_name, "
            "be.pchembl AS pchembl, be.activity_type AS activity_type "
            "FROM bioactivity_evidence be "
            "JOIN compounds c ON c.id = be.compound_id "
            "ORDER BY be.id"
        ),
    },
    "EVIDENCE_FOR_TARGET": {
        "source_table": "bioactivity_evidence",
        "src_type": "BioactivityEvidence",
        "tgt_type": "Target",
        "query": (
            "SELECT be.id AS src_name, "
            "  COALESCE(t.name, be.target_pref_name) AS tgt_name, "
            "  be.assay_confidence AS confidence_score, "
            "  be.publication_year AS year "
            "FROM bioactivity_evidence be "
            "LEFT JOIN targets t ON t.name = be.target_pref_name "
            "ORDER BY be.id"
        ),
    },
    # -- Phase 2 symptom→disease materialized map (audit §4.2) --
    "MAPS_TO_DISEASE": {
        "source_table": "symptom_disease_map",
        "src_type": "Symptom",
        "tgt_type": "Disease",
        # Phase 3 — JOIN through diseases_canonical when a MeSH match exists
        # so the edge points at the canonical Disease entity, not a free-text
        # alias. Falls back to symptom_disease_map.disease_name for rows that
        # don't yet have a canonical anchor.
        # Filter out tier-4 string-match rows (match_score=0.3) — they're
        # noisy and pollute the recommendation graph (#50). Keep tiers 1-3
        # (≥0.5) which represent exact / Jaccard / substring matches.
        "query": (
            "SELECT s.name AS src_name, "
            "       COALESCE(d.preferred_name, sdm.disease_name) AS tgt_name, "
            "       sdm.source AS source, sdm.mesh_id AS mesh_id, "
            "       sdm.umls_id AS umls_id, sdm.icd10cm_id AS icd10cm_id, "
            "       sdm.match_score AS match_score "
            "FROM symptom_disease_map sdm "
            "JOIN symptoms s ON s.id = sdm.symptom_id "
            "LEFT JOIN diseases_canonical d ON d.mesh_id = sdm.mesh_id "
            "WHERE sdm.match_score >= 0.5 "
            "ORDER BY s.id, sdm.match_score DESC"
        ),
    },
    # -- Phase 3 evidence-typed compound→disease relationships --
    "COMPOUND_TREATS_DISEASE": {
        "source_table": "compound_disease_evidence",
        "src_type": "Compound",
        "tgt_type": "Disease",
        "query": (
            "SELECT c.name AS src_name, d.preferred_name AS tgt_name, "
            "       cde.pubmed_ids AS pubmed_ids, cde.source AS source "
            "FROM compound_disease_evidence cde "
            "JOIN compounds c ON c.id = cde.compound_id "
            "JOIN diseases_canonical d ON d.id = cde.disease_id "
            "WHERE cde.evidence_type = 'direct_therapeutic' "
            "ORDER BY cde.id"
        ),
    },
    "COMPOUND_MARKER_FOR_DISEASE": {
        "source_table": "compound_disease_evidence",
        "src_type": "Compound",
        "tgt_type": "Disease",
        "query": (
            "SELECT c.name AS src_name, d.preferred_name AS tgt_name, "
            "       cde.pubmed_ids AS pubmed_ids "
            "FROM compound_disease_evidence cde "
            "JOIN compounds c ON c.id = cde.compound_id "
            "JOIN diseases_canonical d ON d.id = cde.disease_id "
            "WHERE cde.evidence_type = 'direct_marker' "
            "ORDER BY cde.id"
        ),
    },
    "COMPOUND_INFERRED_DISEASE": {
        "source_table": "compound_disease_evidence",
        "src_type": "Compound",
        "tgt_type": "Disease",
        "query": (
            "SELECT c.name AS src_name, d.preferred_name AS tgt_name, "
            "       cde.inference_gene_symbol AS gene, "
            "       cde.inference_score AS score, cde.pubmed_ids AS pubmed_ids "
            "FROM compound_disease_evidence cde "
            "JOIN compounds c ON c.id = cde.compound_id "
            "JOIN diseases_canonical d ON d.id = cde.disease_id "
            "WHERE cde.evidence_type = 'inferred_via_gene' "
            "ORDER BY cde.id"
        ),
    },
    # -- Phase 4 KEGG pathway overlay --
    "COMPOUND_IN_PATHWAY": {
        "source_table": "kegg_compound_pathways",
        "src_type": "Compound",
        "tgt_type": "Pathway",
        # Joins through compound_identity.kegg_compound_id; only fires when
        # Phase 1 ingest has populated compound_identity. Until then this
        # query returns 0 rows — that's acceptable per spec §6.
        "query": (
            "SELECT c.name AS src_name, kp.name AS tgt_name, "
            "       kp.id AS pathway_id "
            "FROM kegg_compound_pathways kcp "
            "JOIN compound_identity ci "
            "    ON ci.kegg_compound_id = kcp.kegg_compound_id "
            "JOIN compounds c ON c.id = ci.compound_id "
            "JOIN kegg_pathways kp ON kp.id = kcp.kegg_pathway_id "
            "ORDER BY c.id, kp.id"
        ),
    },
    "PATHWAY_INCLUDES_TARGET": {
        "source_table": "kegg_pathway_genes",
        "src_type": "Pathway",
        "tgt_type": "Target",
        # Joins kegg_pathway_genes.gene_symbol → targets.gene_symbol.
        # Works without Phase 1 ingest — 455 joins on the live DB today.
        "query": (
            "SELECT kp.name AS src_name, t.name AS tgt_name, "
            "       kpg.kegg_gene_id AS kegg_gene_id, "
            "       kpg.gene_symbol AS gene_symbol "
            "FROM kegg_pathway_genes kpg "
            "JOIN targets t ON t.gene_symbol = kpg.gene_symbol "
            "JOIN kegg_pathways kp ON kp.id = kpg.kegg_pathway_id "
            "ORDER BY kp.id, t.id"
        ),
    },
    # -- Tenant relationship types (clinical practice layer) --
    # These have no SQLite source — ingested via tenant API (Phase 4).
    "INCLUDES": {
        "source_table": None,
        "src_type": "Protocol",
        "tgt_type": "Intervention",
        "query": None,
    },
    "USES": {
        "source_table": None,
        "src_type": "Intervention",
        "tgt_type": "Compound",  # also Herb, Food
        "query": None,
    },
    "RESULTED_IN": {
        "source_table": None,
        "src_type": "Intervention",
        "tgt_type": "Outcome",
        "query": None,
    },
    "MEASURED_BY": {
        "source_table": None,
        "src_type": "Outcome",
        "tgt_type": "Biomarker",
        "query": None,
    },
    "INDICATES": {
        "source_table": None,
        "src_type": "Biomarker",
        "tgt_type": "Disease",  # also Symptom
        "query": None,
    },
    "CONTRAINDICATES": {
        "source_table": None,
        # Polymorphic: covers both the legacy tenant stub (Compound/Intervention
        # → Disease/Symptom) and the HDI-scoped case (Herb → Condition/
        # PatientState) ingested from hdi_safe_50.json + MSK/NIH ODS.
        "src_type": "Herb|Compound|Intervention",
        "tgt_type": "Condition|Disease|Symptom|PatientState",
        "description": (
            "Contraindication: substance should not be used in the target state "
            "(pregnancy, hepatic/renal impairment, pediatric, disease overlap, etc.)"
        ),
        "query": None,
    },
    "SYNERGIZES_WITH": {
        "source_table": None,
        "src_type": "Compound",  # also Intervention → Intervention
        "tgt_type": "Compound",
        "query": None,
    },
    "INTERACTS_WITH": {
        "source_table": None,  # curated via hdi_safe_50.json, ingested by ingest_hdi.py
        "src_type": "Herb",
        "tgt_type": "Drug",
        "description": "Herb-drug interaction (HDI) from NIH ODS / MSK About Herbs / LiverTox",
        "query": None,
    },
}


# ---------------------------------------------------------------------------
# Description generators — produce searchable text for each entity
# ---------------------------------------------------------------------------


def describe_herb(row: dict[str, Any]) -> str:
    """Generate a rich description for an Herb entity."""
    parts = [row.get("scientific_name", "Unknown herb")]
    if row.get("common_name"):
        parts[0] += f" ({row['common_name']})"
    if row.get("family"):
        parts.append(f"Family: {row['family']}")
    if row.get("genus") and row.get("species"):
        parts.append(f"Taxonomy: {row['genus']} {row['species']}")
    if row.get("usage_type"):
        parts.append(f"Use: {row['usage_type']}")
    if row.get("alternate_names"):
        try:
            names = json.loads(row["alternate_names"]) if isinstance(
                row["alternate_names"], str
            ) else row["alternate_names"]
            if names:
                parts.append(f"Also known as: {', '.join(names[:5])}")
        except (json.JSONDecodeError, TypeError):
            pass
    return ". ".join(parts)


def describe_compound(row: dict[str, Any]) -> str:
    """Generate a rich description for a Compound entity."""
    parts = [row.get("name", "Unknown compound")]
    if row.get("compound_class"):
        parts.append(f"Class: {row['compound_class']}")
    if row.get("cas_number"):
        parts.append(f"CAS: {row['cas_number']}")
    if row.get("bioactivities"):
        try:
            activities = json.loads(row["bioactivities"]) if isinstance(
                row["bioactivities"], str
            ) else row["bioactivities"]
            if activities:
                parts.append(f"Bioactivities: {', '.join(activities[:10])}")
        except (json.JSONDecodeError, TypeError):
            pass
    return ". ".join(parts)


def describe_food(row: dict[str, Any]) -> str:
    """Generate a rich description for a Food entity."""
    parts = [row.get("food_name", "Unknown food")]
    if row.get("food_name_scientific"):
        parts[0] += f" ({row['food_name_scientific']})"
    if row.get("food_group"):
        parts.append(f"Group: {row['food_group']}")
    if row.get("nutrition_100g"):
        try:
            n = json.loads(row["nutrition_100g"]) if isinstance(
                row["nutrition_100g"], str
            ) else row["nutrition_100g"]
            if n:
                summary_parts = []
                if n.get("calories") is not None:
                    summary_parts.append(f"{n['calories']} kcal")
                if n.get("protein") is not None:
                    summary_parts.append(f"{n['protein']}g protein")
                if n.get("dietary_fiber") is not None:
                    summary_parts.append(f"{n['dietary_fiber']}g fiber")
                if n.get("vitamin_c") is not None and n["vitamin_c"] > 0:
                    summary_parts.append(f"{n['vitamin_c']}mg vitamin C")
                if n.get("iron") is not None and n["iron"] > 0:
                    summary_parts.append(f"{n['iron']}mg iron")
                if summary_parts:
                    parts.append(f"Per 100g: {', '.join(summary_parts)}")
        except (json.JSONDecodeError, TypeError):
            pass
    return ". ".join(parts)


def describe_target(row: dict[str, Any]) -> str:
    """Generate a rich description for a Target entity."""
    parts = [row.get("name", "Unknown target")]
    if row.get("uniprot_id"):
        parts.append(f"UniProt: {row['uniprot_id']}")
    if row.get("gene_symbol"):
        parts.append(f"Gene: {row['gene_symbol']}")
    if row.get("druggability_status"):
        parts.append(f"Druggability: {row['druggability_status']}")
    return ". ".join(parts)


def describe_disease(row: dict[str, Any]) -> str:
    """Generate a rich description for a Disease entity (Phase 3 canonical).

    Renders preferred name + ontology cross-refs (MeSH/UMLS/ICD-10/HPO).
    Falls back to legacy `disease_name` when called with a pre-Phase-3 row
    shape (defensive — old call sites still work during the migration).
    """
    name = row.get("preferred_name") or row.get("disease_name") or "Unknown disease"
    parts = [name]
    refs: list[str] = []
    if row.get("mesh_id"):
        refs.append(f"MeSH {row['mesh_id']}")
    if row.get("umls_id"):
        refs.append(f"UMLS {row['umls_id']}")
    if row.get("icd10cm_id"):
        refs.append(f"ICD-10-CM {row['icd10cm_id']}")
    if row.get("hpo_id"):
        refs.append(f"HPO {row['hpo_id']}")
    if refs:
        parts.append("(" + ", ".join(refs) + ")")
    return ". ".join(parts)


def describe_symptom(row: dict[str, Any]) -> str:
    """Generate a description for a Symptom entity."""
    parts = [row.get("name", "Unknown symptom")]
    if row.get("symptom_type"):
        parts.append(f"Type: {row['symptom_type']}")
    if row.get("description"):
        parts.append(row["description"])
    return ". ".join(parts)


def describe_bioactivity_evidence(row: dict[str, Any]) -> str:
    """Render a single ChEMBL bioactivity record as a search-rich description.

    Phase 1 drug-bioactive bridge — see ADR 0007. Each row is a measured
    drug-target activity from ChEMBL, joined to our compound universe by
    InChIKey via compound_identity.
    """
    relation = row.get("relation") or "="
    value = row.get("value")
    units = row.get("units") or ""
    activity_type = row.get("activity_type") or "activity"
    target = (
        row.get("target_pref_name")
        or row.get("chembl_target_id")
        or "unknown target"
    )
    organism = row.get("target_organism") or ""
    pchembl = row.get("pchembl")
    confidence = row.get("assay_confidence")
    year = row.get("publication_year")
    doc_id = row.get("chembl_doc_id") or ""
    chembl_compound = row.get("chembl_compound_id") or "?"

    parts: list[str] = [
        (
            f"BioactivityEvidence: {chembl_compound} {relation} "
            f"{value if value is not None else '?'}{units} {activity_type} "
            f"against {target}"
        )
    ]
    if organism:
        parts.append(f" ({organism})")
    extras: list[str] = []
    if pchembl is not None:
        extras.append(f"pChEMBL {pchembl}")
    if confidence is not None:
        extras.append(f"assay confidence {confidence}")
    if year:
        extras.append(f"year {year}")
    if doc_id:
        extras.append(f"doc {doc_id}")
    if extras:
        parts.append("; " + ", ".join(extras))
    return "".join(parts)


def describe_pathway(row: dict[str, Any]) -> str:
    """Generate a description for a KEGG Pathway entity (Phase 4)."""
    name = row.get("name", "Unknown pathway")
    parts = [name]
    extras: list[str] = []
    if row.get("organism"):
        extras.append(f"organism: {row['organism']}")
    if row.get("category"):
        extras.append(f"category: {row['category']}")
    if row.get("source"):
        extras.append(f"source: {row['source']}")
    if row.get("id"):
        extras.append(f"KEGG ID: {row['id']}")
    if extras:
        parts.append("(" + ", ".join(extras) + ")")
    return ". ".join(parts)


# ---------------------------------------------------------------------------
# Tenant entity description generators (clinical practice layer)
# ---------------------------------------------------------------------------


def describe_protocol(row: dict[str, Any]) -> str:
    """Generate a description for a Protocol entity (tenant-scoped)."""
    parts = [row.get("name", "Unknown protocol")]
    if row.get("description"):
        parts.append(row["description"])
    if row.get("phase"):
        parts.append(f"Phase: {row['phase']}")
    if row.get("target_conditions"):
        try:
            conditions = json.loads(row["target_conditions"]) if isinstance(
                row["target_conditions"], str
            ) else row["target_conditions"]
            if conditions:
                parts.append(f"Conditions: {', '.join(conditions[:5])}")
        except (json.JSONDecodeError, TypeError):
            pass
    if row.get("duration"):
        parts.append(f"Duration: {row['duration']}")
    return ". ".join(parts)


def describe_intervention(row: dict[str, Any]) -> str:
    """Generate a description for an Intervention entity (tenant-scoped)."""
    parts = [row.get("name", "Unknown intervention")]
    if row.get("compound"):
        parts.append(f"Compound: {row['compound']}")
    if row.get("route"):
        parts.append(f"Route: {row['route']}")
    if row.get("dosage"):
        parts.append(f"Dosage: {row['dosage']}")
    if row.get("frequency"):
        parts.append(f"Frequency: {row['frequency']}")
    if row.get("form"):
        parts.append(f"Form: {row['form']}")
    return ". ".join(parts)


def describe_outcome(row: dict[str, Any]) -> str:
    """Generate a description for an Outcome entity (tenant-scoped)."""
    parts = [row.get("name", "Unknown outcome")]
    if row.get("observation"):
        parts.append(row["observation"])
    if row.get("direction"):
        parts.append(f"Direction: {row['direction']}")
    if row.get("magnitude"):
        parts.append(f"Magnitude: {row['magnitude']}")
    if row.get("timeframe"):
        parts.append(f"Timeframe: {row['timeframe']}")
    if row.get("condition"):
        parts.append(f"Condition: {row['condition']}")
    return ". ".join(parts)


def describe_biomarker(row: dict[str, Any]) -> str:
    """Generate a description for a Biomarker entity (tenant-scoped)."""
    parts = [row.get("name", "Unknown biomarker")]
    if row.get("category"):
        parts.append(f"Category: {row['category']}")
    if row.get("unit"):
        parts.append(f"Unit: {row['unit']}")
    if row.get("normal_range"):
        parts.append(f"Normal range: {row['normal_range']}")
    if row.get("target_gene"):
        parts.append(f"Gene: {row['target_gene']}")
    return ". ".join(parts)


def describe_interacts_with(row: dict[str, Any]) -> str:
    """Generate a description for a Herb → Drug INTERACTS_WITH edge (HDI)."""
    return (
        f"{row['herb_name']} interacts with {row['drug_name']} "
        f"({row['severity']} severity; mechanism class: {row['mechanism_class']}; "
        f"evidence: {row['evidence_tier']})"
    )


def describe_contraindicates(row: dict[str, Any]) -> str:
    """Generate a description for a CONTRAINDICATES edge.

    Supports both the legacy Compound→Disease tenant stub and the
    HDI-scoped Herb→Condition case. Uses keyword fallback to stay
    polymorphic without branching on edge direction.
    """
    src = row.get("herb_name") or row.get("compound_name") or row.get("intervention_name", "substance")
    tgt = row.get("condition") or row.get("disease_name") or row.get("symptom_name", "state")
    severity = row.get("severity", "unspecified")
    return f"{src} is contraindicated in {tgt} ({severity})"


# ---------------------------------------------------------------------------
# Dynamic query builders (for tables with optional columns)
# ---------------------------------------------------------------------------


def build_food_query(conn: sqlite3.Connection) -> str:
    """Build Food entity query, including nutrition_100g only if it exists.

    Results are sorted by ``food_name`` so ``LIMIT N`` is reproducible.
    """
    cols = [r[1] for r in conn.execute("PRAGMA table_info(compound_foods)").fetchall()]
    if "nutrition_100g" in cols:
        return (
            "SELECT DISTINCT food_name, food_name_scientific, food_group, "
            "nutrition_100g FROM compound_foods ORDER BY food_name"
        )
    return (
        "SELECT DISTINCT food_name, food_name_scientific, food_group "
        "FROM compound_foods ORDER BY food_name"
    )


def build_disease_query(conn: sqlite3.Connection) -> str:
    """Build Disease entity query, handling missing tables.

    Phase 3 promoted Disease to a first-class entity backed by the
    ``diseases_canonical`` registry. Prefer that source when present; fall
    back to the legacy ``target_diseases`` + ``chemical_diseases`` UNION
    only when the canonical table is absent (one-cycle compatibility window
    while the legacy table is being deprecated — see ADR 0008).

    The outer query sorts by ``disease_name`` so ``LIMIT N`` is
    reproducible even though the inner UNION has no intrinsic order.
    """
    def _has(table: str) -> bool:
        return conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table,),
        ).fetchone() is not None

    if _has("diseases_canonical"):
        return (
            "SELECT preferred_name AS disease_name "
            "FROM diseases_canonical "
            "ORDER BY preferred_name"
        )
    parts = []
    for table, col in [("target_diseases", "disease_name"), ("chemical_diseases", "disease_name")]:
        if _has(table):
            parts.append(f"SELECT DISTINCT {col} AS disease_name FROM {table}")
    if not parts:
        return "SELECT 'none' AS disease_name WHERE 0"
    inner = " UNION ".join(parts)
    return f"SELECT * FROM ({inner}) ORDER BY disease_name"


QUERY_BUILDERS = {
    "build_food_query": build_food_query,
    "build_disease_query": build_disease_query,
}


DESCRIPTION_GENERATORS = {
    "Herb": describe_herb,
    "Compound": describe_compound,
    "Food": describe_food,
    "Target": describe_target,
    "Disease": describe_disease,
    "Symptom": describe_symptom,
    # Phase 1 drug-bioactive bridge
    "BioactivityEvidence": describe_bioactivity_evidence,
    # Phase 4 KEGG pathway overlay
    "Pathway": describe_pathway,
    # Tenant entity types (clinical practice layer)
    "Protocol": describe_protocol,
    "Intervention": describe_intervention,
    "Outcome": describe_outcome,
    "Biomarker": describe_biomarker,
}


# ---------------------------------------------------------------------------
# Relationship description generators
# ---------------------------------------------------------------------------


def describe_relationship(rel_type: str, row: dict[str, Any]) -> tuple[str, str]:
    """Generate (description, keywords) for a relationship edge."""
    src = row.get("src_name", "?")
    tgt = row.get("tgt_name", "?")

    if rel_type == "CONTAINS_COMPOUND":
        part = row.get("plant_part", "")
        conc_lo = row.get("concentration_low_ppm")
        conc_hi = row.get("concentration_high_ppm")
        desc = f"{src} contains {tgt}"
        if part:
            desc += f" in {part}"
        if conc_lo is not None and conc_hi is not None:
            desc += f" ({conc_lo}-{conc_hi} PPM)"
        return desc, "herb compound phytochemical contains"

    if rel_type == "FOUND_IN_FOOD":
        val = row.get("content_value")
        unit = row.get("content_unit", "")
        desc = f"{src} found in {tgt}"
        if val is not None:
            desc += f" ({val} {unit})"
        return desc, "compound food dietary source contains"

    if rel_type == "TARGETS_PROTEIN":
        activity = row.get("activity_value")
        atype = row.get("activity_type", "")
        desc = f"{src} targets {tgt}"
        if activity is not None:
            desc += f" ({atype}: {activity})"
        return desc, "compound target protein interaction mechanism"

    if rel_type == "ASSOCIATED_WITH_DISEASE":
        evidence = row.get("evidence", "")
        desc = f"{src} associated with {tgt}"
        if evidence:
            desc += f" (evidence: {evidence})"
        return desc, "target disease association therapeutic"

    if rel_type == "TREATS_SYMPTOM":
        desc = f"{src} treats {tgt}"
        return desc, "herb symptom treatment traditional medicine"

    # -- Phase 1 drug-bioactive bridge (ChEMBL evidence) --

    if rel_type == "HAS_EVIDENCE":
        pchembl = row.get("pchembl")
        atype = row.get("activity_type") or ""
        desc = f"{src} has measured evidence ({atype}"
        if pchembl is not None:
            desc += f", pChEMBL {pchembl}"
        desc += f") in {tgt}"
        return desc, "compound bioactivity evidence chembl measurement"

    if rel_type == "EVIDENCE_FOR_TARGET":
        confidence = row.get("confidence_score")
        year = row.get("year")
        desc = f"{src} reports activity against {tgt}"
        extras: list[str] = []
        if confidence is not None:
            extras.append(f"assay confidence {confidence}")
        if year:
            extras.append(f"year {year}")
        if extras:
            desc += " (" + ", ".join(extras) + ")"
        return desc, "evidence target measurement assay confidence"

    # -- Phase 4 KEGG pathway overlay --

    if rel_type == "COMPOUND_IN_PATHWAY":
        pid = row.get("pathway_id")
        desc = f"{src} participates in pathway {tgt}"
        if pid:
            desc += f" (KEGG {pid})"
        return desc, "compound pathway kegg metabolism mechanism"

    if rel_type == "PATHWAY_INCLUDES_TARGET":
        gene = row.get("gene_symbol") or row.get("kegg_gene_id") or ""
        desc = f"{src} includes target {tgt}"
        if gene:
            desc += f" (gene {gene})"
        return desc, "pathway target gene kegg mechanism"

    if rel_type == "MAPS_TO_DISEASE":
        # Phase 2 — materialized symptom→disease bridge (audit §4.2).
        source = row.get("source", "string_match")
        mesh = row.get("mesh_id")
        score = row.get("match_score")
        desc = f"{src} maps to disease {tgt}"
        details: list[str] = [f"via {source}"]
        if mesh:
            details.append(f"MeSH {mesh}")
        if score is not None:
            details.append(f"score {score}")
        desc += " (" + ", ".join(details) + ")"
        return desc, "symptom disease ontology mesh umls icd"

    # -- Phase 3 evidence-typed compound→disease relationships --

    if rel_type == "COMPOUND_TREATS_DISEASE":
        pubmed = row.get("pubmed_ids") or ""
        n_cites = (
            len([x for x in pubmed.split("|") if x]) if pubmed else 0
        )
        desc = f"{src} therapeutically treats {tgt}"
        if n_cites:
            desc += f" ({n_cites} PubMed citation{'s' if n_cites != 1 else ''})"
        return desc, "compound disease therapeutic treatment evidence pubmed"

    if rel_type == "COMPOUND_MARKER_FOR_DISEASE":
        pubmed = row.get("pubmed_ids") or ""
        n_cites = (
            len([x for x in pubmed.split("|") if x]) if pubmed else 0
        )
        desc = f"{src} is a marker / mechanism for {tgt}"
        if n_cites:
            desc += f" ({n_cites} PubMed citation{'s' if n_cites != 1 else ''})"
        return desc, "compound disease marker mechanism diagnostic pubmed"

    if rel_type == "COMPOUND_INFERRED_DISEASE":
        gene = row.get("gene") or ""
        score = row.get("score")
        desc = f"{src} inferred to associate with {tgt}"
        if gene:
            desc += f" via {gene}"
        if score is not None:
            desc += f" (score {score})"
        return desc, "compound disease inference gene mediated evidence"

    # -- Tenant relationship types (clinical practice layer) --

    if rel_type == "INCLUDES":
        phase = row.get("phase", "")
        order = row.get("order")
        desc = f"{src} includes intervention {tgt}"
        if phase:
            desc += f" ({phase} phase)"
        if order is not None:
            desc += f" [step {order}]"
        return desc, "protocol intervention treatment plan includes"

    if rel_type == "USES":
        route = row.get("route", "")
        dosage = row.get("dosage", "")
        desc = f"{src} uses {tgt}"
        if route:
            desc += f" via {route}"
        if dosage:
            desc += f" ({dosage})"
        return desc, "intervention compound administration uses therapeutic"

    if rel_type == "RESULTED_IN":
        timeframe = row.get("timeframe", "")
        desc = f"{src} resulted in {tgt}"
        if timeframe:
            desc += f" over {timeframe}"
        return desc, "intervention outcome result clinical effect"

    if rel_type == "MEASURED_BY":
        value = row.get("value")
        unit = row.get("unit", "")
        desc = f"{src} measured by {tgt}"
        if value is not None:
            desc += f" ({value} {unit})"
        return desc, "outcome biomarker measurement laboratory"

    if rel_type == "INDICATES":
        evidence_level = row.get("evidence_level", "")
        desc = f"{src} indicates {tgt}"
        if evidence_level:
            desc += f" (evidence: {evidence_level})"
        return desc, "biomarker disease indicator diagnostic marker"

    if rel_type == "CONTRAINDICATES":
        reason = row.get("reason", "")
        severity = row.get("severity", "")
        desc = f"{src} contraindicated with {tgt}"
        if severity:
            desc += f" (severity: {severity})"
        if reason:
            desc += f" — {reason}"
        return desc, "contraindication warning interaction safety adverse"

    if rel_type == "SYNERGIZES_WITH":
        mechanism = row.get("mechanism", "")
        desc = f"{src} synergizes with {tgt}"
        if mechanism:
            desc += f" via {mechanism}"
        return desc, "synergy combination interaction enhancement potentiation"

    return f"{src} relates to {tgt}", rel_type.lower().replace("_", " ")


def evidence_tier_for(rel_type: str, row: dict[str, Any]) -> str:
    """Structured evidence tier for a relationship edge — the chain-tool wire
    property ``r.evidence_tier`` (read by scoped_server ``/traverse``; #233b/T4.2).

    Normalised but source-faithful:
    - a MEASURED assay activity -> ``"assay"`` (TARGETS_PROTEIN's activity_value,
      HAS_EVIDENCE's pchembl);
    - a curated clinical/regulatory layer is passed through VERBATIM so the panel
      can show it (ASSOCIATED_WITH_DISEASE's evidence_layer, e.g. ``"Approved"``,
      ``"Phase 2"``, ``"Investigative"``);
    - present-but-unmeasured -> ``"annotated"``;
    - edges with no structured evidence (symptom/food/contains) -> ``""``.

    HERB2 ASSOCIATED_WITH_DISEASE rows carry their own clinical/experimental/
    traditional tier already and are populated at their build site, not here.
    """
    if rel_type in ("TARGETS_PROTEIN", "HAS_EVIDENCE"):
        measured = (
            row.get("activity_value")
            if rel_type == "TARGETS_PROTEIN"
            else row.get("pchembl")
        )
        return "assay" if measured not in (None, "") else "annotated"
    if rel_type == "ASSOCIATED_WITH_DISEASE":
        return str(row.get("evidence") or "").strip() or "annotated"
    return ""
