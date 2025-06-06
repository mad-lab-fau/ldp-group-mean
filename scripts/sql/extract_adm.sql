WITH filtered_diag AS (SELECT hadm_id, icd_version, icd_code
                       FROM mimiciv_hosp.diagnoses_icd
                       WHERE seq_num = 1)
SELECT adm.subject_id,
       adm.hadm_id,
       adm.admission_type,
       adm.admission_location,
       adm.discharge_location,
       adm.race,
       adm.hospital_expire_flag,
       di.icd_version,
       di.icd_code,
       a.anchor_age,
       a.age,
       a.anchor_year,
       p.gender,
       p.dod
FROM mimiciv_hosp.admissions adm
         JOIN filtered_diag di ON adm.hadm_id = di.hadm_id
         JOIN mimiciv_derived.age a ON adm.hadm_id = a.hadm_id
         JOIN mimiciv_hosp.patients p ON adm.subject_id = p.subject_id;