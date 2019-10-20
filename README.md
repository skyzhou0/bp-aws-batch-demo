# aws-batch-demo

# 0. Inspect the ML model are trying to build.
virtualenv -p python3 batch-env

source batch-env/bin/activate

pip3 install -r requirements.txt

# unblock line 12 (# import matplotlib.pyplot as plt) and line 33 so that we can plot the training data.

# we run the file till line , the the rest of the lines are better to run in the AWS environment.

# I would like you to replace 'your_name' with your own name in line 64, - 'author': 'your_name'.

# 1. Docker build the image. please change the name of the docker image (or tag name).
docker build -t yourdockerimage .

# 2. Push the docker image to docker hub. Note that this requires you already have a docker hub account, and have docker hub configure on your local environment.
# 2.a. If Yes, then login to docker hub from your local environment and run the following command.
docker push yourdockerimage

# 2.b. If No, I would suggest that we head stright to AWS batch, and use the docker image that I have created.
<!-- hsz4docker/aws-batch-ds-demo -->

# 3. AWS Batch.
# 3.1 Go to "Compute environments",
# 3.1.a. select managed "compute environment type".
# 3.1.b. set the environment name as "bp-aws-batch-env" please choice your own name.
# 3.1.c. set the service role as "AWSBatchServiceRole".
# 3.1.d. set the instance role as "ecsInstanceRole".
# 3.1.e. minimum vCPUs as 1, desired vCPUs as 2, maximum vCPUs as 5.
# 3.1.f. select VPC id as vpc-2734f41

# 3.2. Go to "Job queues"
# 3.2.a. Queue name: "bp-aws-batch-queue", pick you own name.
# 3.2.b. Priority set to be "100".
# 3.2.c. For the Connected copute environments for this queue, we will choose "bp-aws-batch-env"
# 3.2.d. Create the job queue.

# 3.3. Go to "Job definition"
# 3.3.a. Queue name: "bp-aws-batch-job-definition01", pick you own name.
# 3.3.b. set Job attempts to be "1".
# 3.3.c. set Excution timeout to be "120".
# 3.3.d. set the Job role be "AWSBatchDemo" (Allows ECS tasks to call AWS services on your behalf. DynamoDB and S3 in particular).
# 3.3.e. Contatiner image be "hsz4docker/aws-batch-ds-demo"
# 3.3.f. Command be "python /src/adaboost_model.py".
# 3.3.g. vCPU to be "1", memory to be "1024"
# 3.3.h. Click create Job Definition.

# 3.4. Jobs
# 3.4.a. Job name: "bp-aws-batch-job-01", pick you own name.
# 3.4.b. Select Job definition be "bp-aws-batch-job-definition01".
# 3.4.c. Select Job queue be "bp-aws-batch-queue".
# 3.4.d. Submit job
# 3.4.e. Go to Dashboard.




