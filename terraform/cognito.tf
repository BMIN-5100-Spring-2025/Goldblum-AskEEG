variable "app_client_callback_urls" {
  description = "List of allowed callback URLs for the Cognito App Client"
  type        = list(string)
  default     = ["http://localhost:5173"]  # TODO: Add deployed URLs later
}

variable "app_client_logout_urls" {
  description = "List of allowed logout URLs for the Cognito App Client"
  type        = list(string)
  default     = ["http://localhost:5173"]  # TODO: Add deployed URLs later
}

# --- Cognito User Pool ---
resource "aws_cognito_user_pool" "user_pool" {
  name = "askeeg-user-pool"

  # Configure password policies, MFA, etc. as needed
  password_policy {
    minimum_length    = 8
    require_lowercase = true
    require_numbers   = true
    require_symbols   = true
    require_uppercase = true
  }

  auto_verified_attributes = ["email"]

  schema {
    name                     = "email"
    attribute_data_type      = "String"
    mutable                  = false
    required                 = true
    developer_only_attribute = false
  }

  tags = {
    Name        = "AskEEG User Pool"
    Environment = "Dev"
  }
}

# --- Cognito User Pool Client ---
resource "aws_cognito_user_pool_client" "app_client" {
  name = "askeeg-app-client"

  user_pool_id = aws_cognito_user_pool.user_pool.id

  generate_secret = false  # Public client for web apps

  allowed_oauth_flows_user_pool_client = true
  allowed_oauth_flows                  = ["code", "implicit"]
  allowed_oauth_scopes                 = ["openid", "email", "profile", "aws.cognito.signin.user.admin"]
  supported_identity_providers         = ["COGNITO"]

  callback_urls = var.app_client_callback_urls
  logout_urls   = var.app_client_logout_urls

  # Explicit auth flows needed for Amplify JS library
  explicit_auth_flows = [
    "ALLOW_USER_SRP_AUTH",
    "ALLOW_REFRESH_TOKEN_AUTH",
    "ALLOW_CUSTOM_AUTH",
    "ALLOW_USER_PASSWORD_AUTH"
  ]
}

# --- Cognito Identity Pool ---
resource "aws_cognito_identity_pool" "identity_pool" {
  identity_pool_name               = "askeeg_identity_pool"
  allow_unauthenticated_identities = false
  allow_classic_flow               = false

  cognito_identity_providers {
    client_id               = aws_cognito_user_pool_client.app_client.id
    provider_name           = aws_cognito_user_pool.user_pool.endpoint
    server_side_token_check = false
  }

  tags = {
    Name        = "AskEEG Identity Pool"
    Environment = "Dev"
  }
}

# --- IAM Role for Authenticated Users ---
resource "aws_iam_role" "cognito_authenticated_role" {
  name = "askeeg-cognito-authenticated-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Effect = "Allow",
        Principal = {
          Federated = "cognito-identity.amazonaws.com"
        },
        Action = "sts:AssumeRoleWithWebIdentity",
        Condition = {
          StringEquals = {
            "cognito-identity.amazonaws.com:aud" = aws_cognito_identity_pool.identity_pool.id
          },
          "ForAnyValue:StringLike" = {
            "cognito-identity.amazonaws.com:amr" = "authenticated"
          }
        }
      }
    ]
  })

  tags = {
    Name        = "AskEEG Cognito Authenticated Role"
    Environment = "Dev"
  }
}

# --- IAM Policy for S3 Uploads ---
resource "aws_iam_policy" "s3_upload_policy" {
  name        = "askeeg-s3-upload-policy"
  description = "Allow users to upload objects to their specific folder in the user uploads bucket"

  policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Effect = "Allow",
        Action = [
          "s3:PutObject",
          "s3:GetObject",
          "s3:ListBucket"
        ],
        Resource = [
          aws_s3_bucket.user_uploads.arn,
          "${aws_s3_bucket.user_uploads.arn}/*"
        ]
      },
    ]
  })
}


# --- Attach S3 Policy to Authenticated Role ---
resource "aws_iam_role_policy_attachment" "authenticated_s3_attachment" {
  role       = aws_iam_role.cognito_authenticated_role.name
  policy_arn = aws_iam_policy.s3_upload_policy.arn
}

# --- Attach Roles to Identity Pool ---
resource "aws_cognito_identity_pool_roles_attachment" "identity_pool_roles" {
  identity_pool_id = aws_cognito_identity_pool.identity_pool.id

  roles = {
    "authenticated" = aws_iam_role.cognito_authenticated_role.arn
  }
}

data "aws_region" "current" {}

# --- Outputs ---
output "cognito_user_pool_id" {
  value       = aws_cognito_user_pool.user_pool.id
  description = "ID of the Cognito User Pool"
}

output "cognito_user_pool_client_id" {
  value       = aws_cognito_user_pool_client.app_client.id
  description = "ID of the Cognito User Pool Client"
}

output "cognito_identity_pool_id" {
  value       = aws_cognito_identity_pool.identity_pool.id
  description = "ID of the Cognito Identity Pool"
}

output "aws_region" {
  value       = data.aws_region.current.name
  description = "AWS Region"
} 