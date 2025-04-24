resource "aws_s3_bucket" "user_uploads" {
  bucket = "bmin5100-askeeg-${data.aws_caller_identity.current.account_id}-user-uploads"

  tags = {
    Name        = "AskEEG User Uploads"
    Environment = "Dev"
  }
}

resource "aws_s3_bucket_acl" "user_uploads_acl" {
  bucket = aws_s3_bucket.user_uploads.id
  acl    = "private"
}

resource "aws_s3_bucket_cors_configuration" "user_uploads_cors" {
  bucket = aws_s3_bucket.user_uploads.bucket

  cors_rule {
    allowed_headers = ["*"]
    allowed_methods = ["PUT", "POST", "GET", "DELETE"]
    allowed_origins = ["*"]  # NOTE: Replace with specific origins in production!
    expose_headers  = ["ETag"]
    max_age_seconds = 3000
  }
}

# Output the bucket name
output "user_uploads_bucket_name" {
  value       = aws_s3_bucket.user_uploads.bucket
  description = "Name of the S3 bucket for user uploads"
} 