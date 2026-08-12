# Terraform Basics

Terraform is an infrastructure-as-code tool that lets you define cloud resources in configuration files and manage their lifecycle with a plan/apply workflow.

`terraform init` downloads the provider plugins needed for your configuration, such as the Google Cloud provider.

`terraform plan` shows what changes Terraform would make without actually applying them, comparing your configuration to the current state.

`terraform apply` actually creates, updates, or destroys resources to match your configuration.

`terraform destroy` tears down every resource Terraform is managing in the current state file. This is useful for demo projects, since it stops billing for anything you spun up.

Terraform tracks resources in a state file, which maps your configuration to real-world resource IDs. Losing the state file makes it hard for Terraform to know what it's managing, which is why state is often stored remotely (e.g. in a GCS bucket) for anything beyond a single-person demo project.

## Common gotchas

Running `terraform apply` twice with the same configuration should be safe and make no changes the second time -- this property is called idempotency, and it's a core design goal of Terraform.

Variables in Terraform can be set via `.tfvars` files, command-line flags, or environment variables prefixed with `TF_VAR_`.
