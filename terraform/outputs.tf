output "ecr_repository_url" {
  value       = aws_ecr_repository.goldblum_askeeg.repository_url
  description = "The URL of the ECR repository"
}

output "ecs_task_definition_arn" {
  value       = aws_ecs_task_definition.goldblum_askeeg.arn
  description = "The ARN of the ECS task definition"
}

output "s3_bucket_name" {
  value       = aws_s3_bucket.goldblum_askeeg.bucket
  description = "The name of the S3 bucket"
}

output "cloudwatch_log_group" {
  value       = aws_cloudwatch_log_group.goldblum_askeeg.name
  description = "The CloudWatch log group for ECS task logs"
}