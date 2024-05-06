# Full Stack Web Application over Kubernetes

## Create a Cluster
1. Create a kind cluster.
2. Verify that the cluster has been created.
```
$ kind create cluster
```

## Containerize and Push the Frontend of the Application
To deploy the application’s frontend, start by containerizing it. Don’t worry about creating a Dockerfile. It has already been created and available for use in the /usercode/elearning folder.

1. Change the directory to /usercode/elearning.
2. Create a Docker image.
3. Push this image to Docker Hub.

```
$ docker build -t my-username/my-image-name:tag .
```

```
$ docker login -u my-username -p my-password
```

```
$ docker push my-username/my-image-name:tag
```

## Deploy the Database
```
$ kubectl apply -f database.yaml
```

## Create a Service for the Database
```
$ kubectl apply -f database-svc.yaml
```

## Deploy the Frontend of the Application
```
$ kubectl apply -f app.yaml
```

## Create a Service for the Front-end
```
$ kubectl apply -f app-svc.yaml
```

## Create a ConfigMap
```
$ kubectl apply -f configmap.yaml
```

## Delete Previous Resources
Execute the kubectl get pods command in the terminal to check the status of the Pods.

Notice the status of database Pods, which is CrashLoopBackOff. The database Pods are crashing because the PostgreSQL image used in this project requires a password for the database to run, which is yet to be set.

```
$ kubectl delete deployments,pods --all
```

## Configure the Database Deployment
```
$ kubectl apply -f database.yaml
```

```
$ kubectl get pods
```

## Configure the Frontend Deployment
```
$ kubectl apply -f app.yaml
```

## Access the Application
```
$ kubectl port-forward svc/app-svc --address 0.0.0.0 31111:3000
```




