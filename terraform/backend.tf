terraform {
  backend "s3" {
    bucket         = "bmin5100-terraform-state"
    key            = "zackgold@seas.upenn.edu-askeeg/terraform.tfstate"
    region         = "us-east-1"
    encrypt        = true
  }
}