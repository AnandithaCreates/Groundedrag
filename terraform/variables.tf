variable "project_id" {
  description = "Your GCP project ID"
  type        = string
}

variable "region" {
  description = "Region to deploy into"
  type        = string
  default     = "us-central1" # cheapest / most free-tier-friendly region
}

variable "service_name" {
  description = "Name of the Cloud Run service"
  type        = string
  default     = "groundedrag"
}

variable "container_image" {
  description = "Full path to the container image, e.g. gcr.io/PROJECT/groundedrag:latest"
  type        = string
}

variable "portkey_api_key" {
  description = "Portkey API key"
  type        = string
  sensitive   = true
}

variable "portkey_virtual_key" {
  description = "Portkey virtual key referencing your stored Groq credential"
  type        = string
  sensitive   = true
}

variable "qdrant_url" {
  description = "Qdrant Cloud cluster URL"
  type        = string
}

variable "qdrant_api_key" {
  description = "Qdrant Cloud API key"
  type        = string
  sensitive   = true
}
