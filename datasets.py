"""Download, verify and load the three datasets as AIF360 BinaryLabelDatasets.

Every dataset is built with AIF360's StandardDataset, using the options of AIF360's own
AdultDataset / GermanDataset / CompasDataset classes except where a comment says otherwise.
Labels are 0/1 with 1 = favourable outcome; the protected attribute is 0/1 with
1 = privileged group. Raw files are downloaded once into data/raw/ and checked against
recorded SHA256 hashes, so a changed or corrupted file is rejected.

  name    rows    protected attribute (privileged = 1)   favourable label (= 1)
  adult   45,222  sex: Male                              income > 50K
  german   1,000  age: > 25 years                        good credit
  compas   6,172  race: Caucasian                        no recidivism within two years
"""
import hashlib
import io
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from aif360.datasets import StandardDataset

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

# Raw files: (url, member inside zip or None, sha256 of the extracted file)
SOURCES = {
    "adult": {
        "adult.data": ("https://archive.ics.uci.edu/static/public/2/adult.zip", "adult.data",
                       "5b00264637dbfec36bdeaab5676b0b309ff9eb788d63554ca0a249491c86603d"),
        "adult.test": ("https://archive.ics.uci.edu/static/public/2/adult.zip", "adult.test",
                       "a2a9044bc167a35b2361efbabec64e89d69ce82d9790d2980119aac5fd7e9c05"),
    },
    "german": {
        "german.data": ("https://archive.ics.uci.edu/static/public/144/statlog+german+credit+data.zip", "german.data",
                        "b21f3d81db8071257d5ff1deaeba1fd4303b62712e6fcc9715c7a86202cb5871"),
    },
    "compas": {
        # ProPublica repository, pinned to commit bafff5d (HEAD on 11 November 2026)
        "compas-scores-two-years.csv": (
            "https://raw.githubusercontent.com/propublica/compas-analysis/"
            "bafff5da3f2e45eca6c2d5055faad269defd135a/compas-scores-two-years.csv", None,
            "c451db85908b2f7fef1d83203bedf6b71ecda0d5af468d82ae62178f91d0cc7d"),
    },
}

DATASETS = ["adult", "german", "compas"]

# Human-readable description used in tables
DESCRIPTION = {
    "adult": {"protected": "sex", "privileged": "male", "unprivileged": "female", "favourable": "income > 50K"},
    "german": {"protected": "age", "privileged": "age > 25", "unprivileged": "age <= 25", "favourable": "good credit"},
    "compas": {"protected": "race", "privileged": "Caucasian", "unprivileged": "not Caucasian",
               "favourable": "no recidivism in two years"},
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def raw_path(name, filename):
    return DATA_DIR / "raw" / name / filename


def download(name):
    """Fetch the raw files of a dataset if missing and verify every file's SHA256."""
    archives = {}
    for filename, (url, member, digest) in SOURCES[name].items():
        path = raw_path(name, filename)
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            if member is None:
                content = urllib.request.urlopen(url, timeout=60).read()
            else:
                if url not in archives:
                    archives[url] = urllib.request.urlopen(url, timeout=60).read()
                content = zipfile.ZipFile(io.BytesIO(archives[url])).read(member)
            path.write_bytes(content)
        actual = sha256(path)
        if actual != digest:
            raise ValueError(f"{path} has SHA256 {actual}, expected {digest}. Delete it and run again.")
    return [raw_path(name, f) for f in SOURCES[name]]


def _favourable(classes):
    return lambda value: 1.0 if value in classes else 0.0


def _adult():
    columns = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation",
               "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week",
               "native-country", "income-per-year"]
    train_path, test_path = download("adult")
    train = pd.read_csv(train_path, header=None, names=columns, skipinitialspace=True, na_values=["?"])
    # adult.test starts with a comment line ("|1x3 Cross validator"); AIF360 skips it the same way.
    test = pd.read_csv(test_path, header=0, names=columns, skipinitialspace=True, na_values=["?"])
    df = pd.concat([test, train], ignore_index=True)  # same order as AIF360's AdultDataset
    # AIF360 AdultDataset defaults, with sex as the only protected attribute; race then becomes
    # an ordinary categorical feature (the variant shown in AIF360's own docstring).
    return StandardDataset(
        df, label_name="income-per-year", favorable_classes=_favourable({">50K", ">50K."}),
        protected_attribute_names=["sex"], privileged_classes=[["Male"]],
        categorical_features=["workclass", "education", "marital-status", "occupation", "relationship",
                              "native-country", "race"],
        features_to_drop=["fnlwgt"], na_values=["?"])


def _german():
    columns = ["status", "month", "credit_history", "purpose", "credit_amount", "savings", "employment",
               "investment_as_income_percentage", "personal_status", "other_debtors", "residence_since",
               "property", "age", "installment_plans", "housing", "number_of_credits", "skill_level",
               "people_liable_for", "telephone", "foreign_worker", "credit"]
    (path,) = download("german")
    df = pd.read_csv(path, sep=" ", header=None, names=columns)
    # AIF360 GermanDataset defaults with age (> 25 privileged, after Kamiran & Calders) as the only
    # protected attribute. Deviation: AIF360 replaces personal_status by a derived 'sex' column;
    # here personal_status stays an ordinary categorical feature, so all 20 original attributes
    # are used, as in Zemel et al.
    return StandardDataset(
        df, label_name="credit", favorable_classes=_favourable({1}),
        protected_attribute_names=["age"], privileged_classes=[lambda x: x > 25],
        categorical_features=["status", "credit_history", "purpose", "savings", "employment",
                              "personal_status", "other_debtors", "property", "installment_plans", "housing",
                              "skill_level", "telephone", "foreign_worker"])


def _compas():
    (path,) = download("compas")
    df = pd.read_csv(path, index_col="id")
    # ProPublica's filtering, as in AIF360 CompasDataset.default_preprocessing
    df = df[(df.days_b_screening_arrest <= 30) & (df.days_b_screening_arrest >= -30) & (df.is_recid != -1)
            & (df.c_charge_degree != "O") & (df.score_text != "N/A")]
    # AIF360 CompasDataset defaults with race as the only protected attribute (sex becomes a
    # categorical feature). Deviation: c_charge_desc (free-text charge, about 400 categories) is
    # dropped; AIF360 one-hot encodes it, which would multiply LFR's parameter count by about 10.
    return StandardDataset(
        df, label_name="two_year_recid", favorable_classes=_favourable({0}),
        protected_attribute_names=["race"], privileged_classes=[["Caucasian"]],
        categorical_features=["sex", "age_cat", "c_charge_degree"],
        features_to_keep=["sex", "age", "age_cat", "race", "juv_fel_count", "juv_misd_count",
                          "juv_other_count", "priors_count", "c_charge_degree", "two_year_recid"])


LOADERS = {"adult": _adult, "german": _german, "compas": _compas}


def load(name):
    """Return (BinaryLabelDataset, protected attribute name)."""
    ds = LOADERS[name]()
    assert ds.favorable_label == 1.0 and ds.unfavorable_label == 0.0
    assert set(np.unique(ds.labels)) <= {0.0, 1.0}
    assert set(np.unique(ds.protected_attributes)) <= {0.0, 1.0}
    return ds, ds.protected_attribute_names[0]


if __name__ == "__main__":
    for name in DATASETS:
        ds, pa = load(name)
        s = ds.protected_attributes.ravel()
        print(f"{name}: {ds.features.shape[0]} rows, {ds.features.shape[1]} features after one-hot encoding, "
              f"privileged share {s.mean():.3f}, favourable share {ds.labels.mean():.3f}")
