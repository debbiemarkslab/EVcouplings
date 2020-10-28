# EC2 Instance (compute)
# To add more VMs, copy/paste this block
resource "aws_instance" "evcouplings_ec2" {
  ami                  = "ami-03657b56516ab7912"
  instance_type        = "t2.micro"
  iam_instance_profile = aws_iam_instance_profile.evcouplings_profile.name

  tags = {
    created-by = "terraform"
  }
}

# S3 Bucket (storage)
resource "aws_s3_bucket" "evcouplings_s3" {
  bucket = "evcouplings-configs-and-results"
  acl    = "public-read"

  tags = {
    Name       = "EVCouplings Bucket"
    created-by = "terraform"
  }
}

# IAM Role (security access object)
resource "aws_iam_role" "evcouplings_ec2_role" {
  name = "evcouplings_ec2_role"

  assume_role_policy = <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Action": "sts:AssumeRole",
      "Principal": {
        "Service": "ec2.amazonaws.com"
      },
      "Effect": "Allow",
      "Sid": ""
    }
  ]
}
EOF

  tags = {
    created-by = "terraform"
  }
}

# IAM Profile (to assign a role to a resource such as an EC2 instance)
resource "aws_iam_instance_profile" "evcouplings_profile" {
  name = "evcouplings_profile"
  role = aws_iam_role.evcouplings_ec2_role.name
}

# IAM Policy (modular access rules to give to Roles)
resource "aws_iam_policy" "evcouplings_s3_read_policy" {
  name        = "evcouplings_s3_bucket_read_policy"
  description = "A policy allowing read access to the evcouplings S3 bucket"

  policy = <<EOF
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "s3:Get*",
                "s3:List*"
            ],
            "Resource": "${aws_s3_bucket.evcouplings_s3.arn}"
        }
    ]
}
EOF
}

# Attach the policy to the role
resource "aws_iam_role_policy_attachment" "evcouplings_attach" {
  role       = aws_iam_role.evcouplings_ec2_role.name
  policy_arn = aws_iam_policy.evcouplings_s3_read_policy.arn
}
