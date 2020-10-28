variable "aws_region" {}
variable "aws_access_key_id" {}
variable "aws_secret_access_key" {}
variable "aws_zones" {
  type        = list(string)
  description = "List of availability zones to use"
  default     = ["us-east-2"]
}
