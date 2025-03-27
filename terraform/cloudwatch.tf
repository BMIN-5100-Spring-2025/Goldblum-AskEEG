resource "aws_cloudwatch_log_group" "goldblum_askeeg" {
  name              = "/ecs/goldblum-askeeg"
  retention_in_days = 30

  tags = {
    Name  = "goldblum-askeeg"
    Owner = element(split("/", data.aws_caller_identity.current.arn), 1)
  }
} 