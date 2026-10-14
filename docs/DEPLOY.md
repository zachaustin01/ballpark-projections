# Deploying to Google Cloud (Firebase Hosting + GitHub Actions)

**Architecture:** GitHub Actions (free for public repos) runs the Python batch job on a cron, builds the static
site, and deploys it to Firebase Hosting (Google's CDN). No servers, no database, no stored credentials.

**Cost / safety:** Stay on the free **Spark** plan: 10 GB storage, ~10 GB/month transfer, and *no billing
account attached*, so you cannot be charged. If traffic outgrows it, the site pauses instead of billing you.
When you want more, upgrade to Blaze (~$0.15/GB egress) and set a budget alert first.
(A plain GCS bucket needs a load balancer for HTTPS, about $18/month, so it is not cheaper.)

## One-time setup (run in Cloud Shell or with gcloud; replace the placeholders)

```bash
PROJECT_ID=your-unique-project-id
gcloud projects create $PROJECT_ID
gcloud config set project $PROJECT_ID
gcloud services enable firebasehosting.googleapis.com iamcredentials.googleapis.com sts.googleapis.com serviceusage.googleapis.com

# Add Firebase to the project (console.firebase.google.com -> Add project -> pick the existing project, Spark plan)
# and create the hosting site: Hosting -> Get started.

# Keyless GitHub -> GCP auth, locked to this repo only
gcloud iam service-accounts create gh-deployer
SA=gh-deployer@$PROJECT_ID.iam.gserviceaccount.com
gcloud projects add-iam-policy-binding $PROJECT_ID --member=serviceAccount:$SA --role=roles/firebasehosting.admin
gcloud projects add-iam-policy-binding $PROJECT_ID --member=serviceAccount:$SA --role=roles/serviceusage.serviceUsageConsumer

gcloud iam workload-identity-pools create github --location=global
gcloud iam workload-identity-pools providers create-oidc github-oidc --location=global \
  --workload-identity-pool=github --issuer-uri=https://token.actions.githubusercontent.com \
  --attribute-mapping=google.subject=assertion.sub,attribute.repository=assertion.repository \
  --attribute-condition="assertion.repository=='zachaustin01/ballpark-projections'"

PROJECT_NUM=$(gcloud projects describe $PROJECT_ID --format='value(projectNumber)')
gcloud iam service-accounts add-iam-policy-binding $SA --role=roles/iam.workloadIdentityUser \
  --member="principalSet://iam.googleapis.com/projects/$PROJECT_NUM/locations/global/workloadIdentityPools/github/attribute.repository/zachaustin01/ballpark-projections"
```

Then set three **repository variables** (Settings -> Secrets and variables -> Actions -> Variables; none are secret):

| Variable | Value |
|---|---|
| `GCP_PROJECT_ID` | your project id |
| `GCP_DEPLOY_SA` | `gh-deployer@<project>.iam.gserviceaccount.com` |
| `GCP_WIF_PROVIDER` | `projects/<number>/locations/global/workloadIdentityPools/github/providers/github-oidc` |

Optionally add a $1 budget alert (Billing -> Budgets) if you ever attach a billing account.

## Notes
- Only today + tomorrow are exported each run, so the site shows a rolling window (no history yet).
- Deploys happen on push to `main`, on the 4x-daily schedule, and manually (Actions -> Run workflow).
- Pull requests only run the build check.
