# terraform {
#  backend "s3" {
#    bucket         = "example-terraform-environmentname"
#    key            = "shared/terraform.tfstate"
#    region         = "us-east-2"
#    encrypt        = true
#    dynamodb_table = "terraform-lock"
#  }
# }
