resource "aws_ecr_repository" "goldblum_askeeg" {
  name                 = "goldblum_askeeg"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }

  tags = {
    Name  = "goldblum_askeeg"
    Owner = element(split("/", data.aws_caller_identity.current.arn), 1)
  }
} 