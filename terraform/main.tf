# No GCS bucket in this version -- Qdrant Cloud hosts the vector index,
# so there's nothing for this project to store in GCS. Cloud Run is the
# only GCP resource left to provision.

resource "google_cloud_run_v2_service" "groundedrag" {
  name     = var.service_name
  location = var.region
  ingress  = "INGRESS_TRAFFIC_ALL"

  template {
    containers {
      image = var.container_image

      env {
        name  = "GROQ_MODEL"
        value = "llama-3.3-70b-versatile"
      }
      env {
        name  = "PORTKEY_API_KEY"
        value = var.portkey_api_key
      }
      env {
        name  = "PORTKEY_VIRTUAL_KEY"
        value = var.portkey_virtual_key
      }
      env {
        name  = "QDRANT_URL"
        value = var.qdrant_url
      }
      env {
        name  = "QDRANT_API_KEY"
        value = var.qdrant_api_key
      }

      resources {
        limits = {
          # Small limits on purpose: keeps the service comfortably inside
          # the Cloud Run always-free monthly compute allowance.
          cpu    = "1"
          memory = "512Mi"
        }
      }
    }

    scaling {
      min_instance_count = 0 # scales to zero -- no cost while idle
      max_instance_count = 2 # capped low; this is a demo, not a product
    }
  }
}

# Public access so you (or an interviewer) can open the URL directly.
# Fine for a read-only demo assistant with no user data involved --
# would NOT be the right default for anything handling real user data.
resource "google_cloud_run_v2_service_iam_member" "public_invoker" {
  location = google_cloud_run_v2_service.groundedrag.location
  name     = google_cloud_run_v2_service.groundedrag.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}
