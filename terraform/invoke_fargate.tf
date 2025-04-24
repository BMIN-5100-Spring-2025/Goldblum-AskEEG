locals {
  container_name = "goldblum-askeeg"
}

module "invoke_fargate_lambda" {
  source = "github.com/BMIN-5100-Spring-2025/infrastructure.git//invoke_fargate_lambda/terraform?ref=f844e9c04f901768ccb99aff77286165bf71b83e"

  project_name = "goldblum-askeeg"
  ecs_task_definition_arn = aws_ecs_task_definition.goldblum_askeeg.arn
  ecs_task_execution_role_arn = aws_iam_role.ecs_task_execution_role.arn
  ecs_task_task_role_arn = aws_iam_role.ecs_task_role.arn
  ecs_task_definition_container_name = local.container_name

  ecs_cluster_arn = data.terraform_remote_state.infrastructure.outputs.ecs_cluster_arn
  ecs_security_group_id = data.terraform_remote_state.infrastructure.outputs.ecs_security_group_id
  private_subnet_id = data.terraform_remote_state.infrastructure.outputs.private_subnet_id
  api_gateway_authorizer_id = data.terraform_remote_state.infrastructure.outputs.api_gateway_authorizer_id
  api_gateway_execution_arn = data.terraform_remote_state.infrastructure.outputs.api_gateway_execution_arn
  api_gateway_id = data.terraform_remote_state.infrastructure.outputs.api_gateway_id
  environment_variables = {}
} 
