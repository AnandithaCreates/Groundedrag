# Cloud Run Basics

Cloud Run is a managed compute platform that runs stateless containers. You give it a container image, and it handles scaling, load balancing, and HTTPS for you.

Cloud Run scales to zero when there's no traffic, which means you pay nothing while idle. It also scales up automatically when requests increase, up to a configurable maximum instance count.

Each Cloud Run service has a concurrency setting, which controls how many requests a single container instance can handle at once. The default concurrency is 80.

Cloud Run has an always-free tier: 2 million requests per month, 360,000 GB-seconds of memory, and 180,000 vCPU-seconds of compute, per month, forever -- not just during a trial period.

Deployments to Cloud Run can be done via `gcloud run deploy`, via Cloud Build triggers, or via Terraform using the `google_cloud_run_v2_service` resource.

## Common gotchas

Cloud Run services are stateless. Anything written to the container's local filesystem is lost when the container restarts or scales down, so persistent data needs to go to Cloud Storage, a database, or another external store.

Cold starts happen when a new container instance has to spin up to handle a request after scaling to zero. This adds latency to the first request after an idle period.
