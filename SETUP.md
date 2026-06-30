# Setup Guide: Remote Corpus Filter

This guide walks through the one-time setup so collaborators can run the
corpus filter from GitHub's web interface and get results in a Google Sheet.

## What the collaborator does (after setup)

1. Edit the word list in the [input Google Sheet](https://docs.google.com/spreadsheets/d/1qhmuSIxo_kpRudFiyVenyr4BsvoNH4NVoVbzJELnQ5g)
2. Go to the GitHub repo → **Actions** tab → **Filter Corpus by Category**
3. Click **Run workflow**, pick a category (and optionally a gender filter)
4. Wait ~2 minutes — results appear as a new tab in the output Google Sheet

## One-time setup (admin)

### Step 1: Create a Google Cloud service account (~10 min)

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project (e.g. "ravensbruck-pipeline") — no billing needed
3. In the left sidebar: **APIs & Services → Library**
4. Search for **Google Sheets API** and click **Enable**
5. In the left sidebar: **APIs & Services → Credentials**
6. Click **Create Credentials → Service Account**
7. Name it anything (e.g. "sheet-writer"), click through the rest
8. Click on the newly created service account
9. Go to the **Keys** tab → **Add Key → Create new key → JSON**
10. A JSON file downloads — keep it safe, you'll need its contents in Step 3

### Step 2: Share the output Google Sheet

1. Create a new Google Sheet for results (or use an existing one)
2. Copy its ID from the URL (the long string between `/d/` and `/edit`)
3. Click **Share** and add the service account email
   (looks like `sheet-writer@ravensbruck-pipeline.iam.gserviceaccount.com`)
4. Give it **Editor** access

### Step 3: Add secrets to the GitHub repo

1. Go to the GitHub repo → **Settings → Secrets and variables → Actions**
2. Add two **Repository secrets**:

| Secret name          | Value                                              |
|---------------------|----------------------------------------------------|
| `GOOGLE_CREDENTIALS` | Paste the **entire contents** of the JSON key file |
| `OUTPUT_SHEET_ID`    | The Google Sheet ID from Step 2                    |

### Step 4: Push the parquet file with Git LFS

Git LFS is already configured. From your local machine:

```bash
git add .gitattributes data/metadata.parquet
git commit -m "Add parquet data via LFS"
git push
```

### Step 5: Update workflow categories (when the sheet changes)

If new columns are added to the input Google Sheet, update the category
list in `.github/workflows/filter_corpus.yml` under `inputs.category.options`.
