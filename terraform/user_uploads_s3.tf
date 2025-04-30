resource "aws_s3_bucket" "user_uploads" {
  bucket = "bmin5100-askeeg-${data.aws_caller_identity.current.account_id}-user-uploads"

  tags = {
    Name        = "AskEEG User Uploads"
    Environment = "Dev"
  }
}

resource "aws_s3_bucket_cors_configuration" "user_uploads_cors" {
  bucket = aws_s3_bucket.user_uploads.id

  cors_rule {
    allowed_headers = ["*"]
    allowed_methods = ["GET", "POST", "PUT", "HEAD"]
    allowed_origins = ["http://localhost:3000", "http://localhost:5173", "bmin-5100.com", "*.bmin-5100"]
    expose_headers  = ["ETag"]
    max_age_seconds = 3000
  }
}

# Output the bucket name
output "user_uploads_bucket_name" {
  value       = aws_s3_bucket.user_uploads.bucket
  description = "Name of the S3 bucket for user uploads"
} 