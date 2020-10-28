## Prerequisites
### Terraform Installed
- Follow your desired set of instructions from the [terraform site](https://www.terraform.io/downloads.html)
### AWS Configured (for AWS users)
- If you are going to run terraform with AWS, you will first need to have an AWS account and have the AWS CLI installed.
- Then, configure your AWS credentials locally. To do so, run `aws configure` and follow the prompts.

## Running Terraform
- In the directory of your corresponding cloud providor (e.g. infra/aws) first run `terraform init` to begin working with Terraform with this project.
- Make any changes to the existing files as desired. For help making changes, see the "Understanding TF Files" section.
- Set any necessary Terraform environment variables. There should be one corresponding with any variable in `vars.tf` that does not have a default value. This will include at least the aws_access_key_id, aws_secret_access_key, and the aws_region. To read more about this, see the "Understanding TF Files" section.
- To make sure the files are still valid, run `terraform validate`.
- To see what cloud resources this will spin up, you can now run `terraform plan`.
- You can now run `terraform apply` to deploy your cloud infrastructure!

## Understanding TF Files
The Terraform files are sectioned into somewhat sensible chunks. The ones that you will probably touch most are `vars.tf` and `main.tf`.We will, however, also briefly elaborate on the others as well.
### vars.tf
- This will hold any values that can be used in other .tf files, some with values defined and some without. The ones without values defined are ones that are secret and can be ingested at runtime. The ones with values simply give you the flexibility to define reusable values.
- Secret variables (values that just have empty brackets) are most easily set using environment variables. To do so, you can export environment variables with the prefix `TF_VAR_` and the following string will be set to that value in vars.tf
    - For example, TF_VAR_aws_region will correspond with aws_region in your vars.tf
    - You can also simply supply the values at runtime, as Terraform will prompt you for any missing values
### main.tf
- This is where you will define your resources such as any compute machines and storage buckets. Please see the provided examples for guidance.
### backend.tf
- Terraform requires a backend to store the state of the deployed environment. It defaults to using a local state file, but if you will be collaborating with others, it may be worth your time to look into backend files. A commented out example is included as backend.tf. To read more about backends, please see the [documentation](https://www.terraform.io/docs/backends/index.html).
### providers.tf
- This is where we provide any necessary cloud provider credentials and other provider-related information.
### outputs.tf
- Here you can define any outputs whose values may not be known until infrastructure has been provisioned (e.g. a dynamically allocated public IP address of a compute node). These are useful for getting information about the runtime environment which can be then passed to any applications (such as evcouplings).
