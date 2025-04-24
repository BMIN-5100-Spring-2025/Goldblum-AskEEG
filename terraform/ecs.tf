resource "aws_ecs_task_definition" "goldblum_askeeg" {
  family                   = "goldblum-askeeg"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = 1024
  memory                   = 2048
  execution_role_arn       = aws_iam_role.ecs_task_execution_role.arn
  task_role_arn            = aws_iam_role.ecs_task_role.arn

  container_definitions = jsonencode([
    {
      name         = "goldblum-askeeg"
      image        = "${aws_ecr_repository.goldblum_askeeg.repository_url}:latest"
      essential    = true
      
      environment = [
        {
          name  = "S3_BUCKET_ARN"
          value = aws_s3_bucket.goldblum_askeeg.arn
        },
        {
          name  = "S3_BUCKET_NAME"
          value = aws_s3_bucket.goldblum_askeeg.bucket
        },
        {
          name  = "INPUT_DIR"
          value = "/data/input"
        },
        {
          name  = "OUTPUT_DIR"
          value = "/data/output"
        },
        {
          name  = "RUN_MODE"
          value = "fargate"
        }
      ]
      
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.goldblum_askeeg.name
          "awslogs-region"        = data.aws_region.current_region.name
          "awslogs-stream-prefix" = "ecs"
        }
      }
      
      mountPoints = [
        {
          sourceVolume  = "data-volume"
          containerPath = "/data"
          readOnly      = false
        }
      ]
      
      portMappings = [
        {
          containerPort = 8000
          hostPort      = 8000
          protocol      = "tcp"
        }
      ]
    }
  ])

  volume {
    name = "data-volume"
  }

  ephemeral_storage {
    size_in_gib = 25
  }
} 