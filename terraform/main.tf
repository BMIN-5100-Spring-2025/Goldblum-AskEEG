resource "aws_s3_bucket" "goldblum_askeeg" {
  bucket = "goldblum-askeeg"

  tags = {
    Owner = element(split("/", data.aws_caller_identity.current.arn), 1)
  }
}

resource "aws_s3_bucket_ownership_controls" "goldblum_askeeg_ownership_controls" {
  bucket = aws_s3_bucket.goldblum_askeeg.id
  rule {
    object_ownership = "BucketOwnerPreferred"
  }
}

resource "aws_s3_bucket_acl" "goldblum_askeeg_acl" {
  depends_on = [aws_s3_bucket_ownership_controls.goldblum_askeeg_ownership_controls]

  bucket = aws_s3_bucket.goldblum_askeeg.id
  acl    = "private"
}

resource "aws_s3_bucket_lifecycle_configuration" "goldblum_askeeg_expiration" {
  bucket = aws_s3_bucket.goldblum_askeeg.id

  rule {
    id  	= "compliance-retention-policy"
    status  = "Enabled"

    expiration {
	  days = 100
    }
  }
}