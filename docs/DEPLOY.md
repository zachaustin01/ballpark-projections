# Architecture, deployment and cost

Live site: https://ballpark-projections-za.web.app

## Architecture

```
GitHub Actions (cron 4x/day + push to main + manual)
  1. restore Statcast store from Actions cache      data/statcast/  (~11 MB, 1 parquet per game day)
  2. ballpark export  -> sync new Statcast days, project each game -> web/public/data/**.json
  3. npm run build    -> web/dist (static React site + the JSON)
  4. keyless auth to Google Cloud, firebase deploy --only hosting
                                   |
                                   v
Firebase Hosting (Google CDN) --> visitors fetch static files only; no server, no database
```

| Piece | Where it runs | Notes |
|---|---|---|
| Batch job (`ballpark export`) | GitHub Actions, `.github/workflows/publish.yml` | Free: standard runners are unlimited for public repos |
| Statcast store | `data/statcast/`, kept between runs in the Actions cache | Cache key is the UTC date (one save/day); if evicted, the next run re-downloads a year in ~30 s |
| Front end | `web/` (Vite + React + TS), built to static files | ~50 KB gzipped JS; talks to nothing but its own JSON |
| Hosting | Firebase Hosting, project `ballpark-projections-za` | Free Spark plan, no billing account |
| Deploy auth | Workload Identity Federation (no stored keys) | Only workflows from `zachaustin01/ballpark-projections` can use it |

Schedule (UTC): 14:00, 17:00, 20:00, 23:00. Each run exports today + tomorrow, so the site shows a rolling
two-day window (no history yet).

## What is configured (as of 2026-10-14)

- GCP project `ballpark-projections-za` (number `49246114083`), owner `zachaustin01@gmail.com`, **billing: disabled**.
- Firebase added to the project; Hosting site `ballpark-projections-za`.
- Service account `gh-deployer@ballpark-projections-za.iam.gserviceaccount.com` with only
  `roles/firebasehosting.admin` and `roles/serviceusage.serviceUsageConsumer`.
- Workload identity pool `github`, provider `github-oidc`, attribute condition
  `assertion.repository=='zachaustin01/ballpark-projections'`.
- Repo variables (identifiers, not secrets): `GCP_PROJECT_ID`, `GCP_DEPLOY_SA`, `GCP_WIF_PROVIDER`.
- `main` is protected: PR required, `build-and-deploy` check must pass, no force-push or deletion, applies to admins.
- Actions hardening: fork-PR runs need approval, default workflow token is read-only. Pull requests run the
  build only and never deploy.

## Cost

**Expected cost: $0.** The project has no billing account, so Google cannot charge it; hitting a free limit
stops usage instead of billing you. A card on another project (e.g. `sewer-labs-analytics`) does not apply here.
Verify any time:

```bash
gcloud billing projects describe ballpark-projections-za --format="value(billingEnabled)"   # must print False
```

| Item | Plan | Measured / expected use | Cost |
|---|---|---|---|
| GitHub Actions (public repo) | free | ~1-2 min/run (first deploy took 74 s) x 4/day | $0 |
| Actions cache | free, 10 GB per repo | ~6 MB Statcast store (compressed) + ~120 MB npm/uv caches | $0 |
| Firebase Hosting storage | Spark, 10 GB | 1 release kept (~0.15 MB at 2 postseason games; a full 15-game regular-season slate is a few MB) | $0 |
| Firebase Hosting transfer | Spark, ~360 MB/day | ~87 KB for the first game a visitor opens (50 KB app + 37 KB game data, gzipped), ~37 KB per extra game; app JS is cached by the browser | $0 up to roughly 2,000-4,000 visits/day |
| Baseball Savant / MLB Stats API | public, no key | ~1 day of Statcast per run | $0 |
| Cloud Storage, Cloud Run, databases | not used | | $0 |

Free-tier numbers come from Firebase's [usage and pricing page](https://firebase.google.com/docs/hosting/usage-quotas-pricing);
third-party summaries disagree on some figures (storage 5 vs 10 GB, transfer per day vs per month), so check that
page before relying on the limits. Overage rates on Blaze (~$0.15/GB transfer, ~$0.026/GB storage) are also
from secondary sources.

**What happens at the limits (Spark):** when Hosting storage is full, new deploys fail (the live site keeps
serving the last release). If daily transfer is exceeded, expect the site to stop serving until the quota resets.
Neither produces a bill.

### Release history
Firebase keeps old releases for rollback and counts them against the 10 GB storage, and the live channel keeps
**all of them by default**. We don't need rollback (revert in git and redeploy instead), so the live channel is set
to keep only the current release (done 2026-10-14):

```bash
curl -X PATCH -H "Authorization: Bearer $(gcloud auth print-access-token)" \
  -H "x-goog-user-project: ballpark-projections-za" -H "Content-Type: application/json" \
  "https://firebasehosting.googleapis.com/v1beta1/projects/ballpark-projections-za/sites/ballpark-projections-za/channels/live?updateMask=retainedReleaseCount" \
  -d '{"retainedReleaseCount":1}'
```
Console: https://console.firebase.google.com/project/ballpark-projections-za/hosting/sites/ballpark-projections-za
See [Manage hosting resources](https://firebase.google.com/docs/hosting/manage-hosting-resources).

### If traffic outgrows the free plan
Upgrading means attaching a billing account (Blaze), which makes costs possible. If you do:
1. Set a budget alert first. Budgets only **alert**, they do not cap spend.
2. Keep response sizes small (the game JSON is ~37 KB gzipped) and leave the CDN cache headers in `firebase.json`.
3. At the measured sizes, 100k game views/month is ~9 GB, inside the 10 GB free month on Blaze.

### Optional: durable history / Statcast store in Cloud Storage
Not needed today. A GCS bucket requires a billing account, so it ends the "can't be charged" guarantee.
It would cost pennies per month. Only worth it if you want multi-day history or can't tolerate cache eviction.

## Security (public repo)
- No secrets in the repo or its history (scanned); deploys use short-lived, keyless Google tokens.
- Repo variables are identifiers only; they cannot be used to sign in.
- The deploy identity can only deploy Hosting in this one project, and only for workflows from this repo.
- Fork pull requests get no Google credentials, and the deploy steps are skipped for all PRs.
- Residual risk: anyone given write access could push a branch and deploy from it. To narrow it, change the
  provider's attribute condition to also require `assertion.ref=='refs/heads/main'` (this blocks deploys from
  other branches, including manual runs from them).

## Operations

```bash
# local: data + UI
uv run ballpark export                 # writes web/public/data/ (auto-syncs Statcast into data/statcast/)
cd web && npm install && npm run dev   # http://localhost:5173

# GitHub
gh run list --repo zachaustin01/ballpark-projections
gh workflow run publish.yml --repo zachaustin01/ballpark-projections   # manual deploy
```

Local runs use the local `data/statcast/` folder; the Actions cache is separate and only exists inside Actions runs.
Changes to `main` go through pull requests. Merging redeploys.

## Reproducing the one-time setup

```bash
PROJECT_ID=ballpark-projections-za REPO=zachaustin01/ballpark-projections
gcloud projects create $PROJECT_ID --name="Ballpark Projections"
gcloud services enable firebase.googleapis.com firebasehosting.googleapis.com iamcredentials.googleapis.com \
  sts.googleapis.com serviceusage.googleapis.com iam.googleapis.com --project $PROJECT_ID
# Add Firebase (console "Add project" on the existing project, Spark plan, no billing) or:
#   POST https://firebase.googleapis.com/v1beta1/projects/$PROJECT_ID:addFirebase

gcloud iam service-accounts create gh-deployer --project $PROJECT_ID
SA=gh-deployer@$PROJECT_ID.iam.gserviceaccount.com
for r in roles/firebasehosting.admin roles/serviceusage.serviceUsageConsumer; do
  gcloud projects add-iam-policy-binding $PROJECT_ID --member=serviceAccount:$SA --role=$r --condition=None; done
gcloud iam workload-identity-pools create github --location=global --project $PROJECT_ID
gcloud iam workload-identity-pools providers create-oidc github-oidc --location=global --project $PROJECT_ID \
  --workload-identity-pool=github --issuer-uri=https://token.actions.githubusercontent.com \
  --attribute-mapping=google.subject=assertion.sub,attribute.repository=assertion.repository \
  --attribute-condition="assertion.repository=='$REPO'"
N=$(gcloud projects describe $PROJECT_ID --format='value(projectNumber)')
gcloud iam service-accounts add-iam-policy-binding $SA --project $PROJECT_ID --role=roles/iam.workloadIdentityUser \
  --member="principalSet://iam.googleapis.com/projects/$N/locations/global/workloadIdentityPools/github/attribute.repository/$REPO"

gh variable set GCP_PROJECT_ID --repo $REPO --body "$PROJECT_ID"
gh variable set GCP_DEPLOY_SA  --repo $REPO --body "$SA"
gh variable set GCP_WIF_PROVIDER --repo $REPO --body "projects/$N/locations/global/workloadIdentityPools/github/providers/github-oidc"
```
