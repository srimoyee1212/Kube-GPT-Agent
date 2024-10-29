from flask import Flask, request, jsonify
from kubernetes import client, config
import os
from typing import Dict, Any, List, Tuple
import logging
from contextlib import contextmanager
import openai
import json

logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s %(levelname)s - %(message)s',
                    filename='agent.log', filemode='a')
logger = logging.getLogger(__name__)

app = Flask(__name__)


#openai.api_key = ''
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    logger.error("OpenAI API key is not set. Please set the OPENAI_API_KEY environment variable.")
    raise Exception("OpenAI API key is not configured")

try:
    config.load_kube_config(os.path.expanduser('~/.kube/config'))
    v1 = client.CoreV1Api()
    apps_v1 = client.AppsV1Api()
except Exception as e:
    logger.error(f"Failed to load kubeconfig: {e}")
    raise

@contextmanager
def k8s_error_handling():
    """Context manager for handling Kubernetes API errors"""
    try:
        yield
    except client.rest.ApiException as e:
        logger.error(f"Kubernetes API error: {e}")
        raise Exception(f"Kubernetes API error: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise Exception(f"Internal server error: {str(e)}")

class KubernetesQueryProcessor:
    def __init__(self):
        self.v1 = client.CoreV1Api()
        self.apps_v1 = client.AppsV1Api()
        
        
        self.operations = {
            "get_pod_status": {
                "description": "Get the status of a specific pod",
                "params": ["pod_name"],
                "example": "What is the status of pod 'nginx'?"
            },
            "list_pods": {
                "description": "List all pods in the default namespace",
                "params": [],
                "example": "List all pods in the default namespace"
            },
            "get_pod_logs": {
                "description": "Get logs from a specific pod",
                "params": ["pod_name"],
                "example": "Show me logs for pod 'nginx'"
            },
            "get_pods_by_deployment": {
                "description": "List all pods created by a specific deployment",
                "params": ["deployment_name"],
                "example": "Which pods are created by deployment 'nginx'?"
            },
            "list_nodes": {
                "description": "List all nodes in the cluster",
                "params": [],
                "example": "List all nodes in the cluster"
            },
            "list_services": {
                "description": "List all services in the default namespace",
                "params": [],
                "example": "List all services in the default namespace"
            },
            "list_deployments": {
                "description": "List all deployments in the default namespace",
                "params": [],
                "example": "List all deployments in the default namespace"
            },
            "count_running_pods": {
                "description": "Count running pods in the default namespace",
                "params": [],
                "example": "How many pods are running in the default namespace?"
            },
            "count_nodes": {
                "description": "Count total nodes in the cluster",
                "params": [],
                "example": "How many nodes are in the cluster?"
            },
            "list_namespaces": {
                "description": "List all namespaces",
                "params": [],
                "example": "List all namespaces"
            }
        }

    def get_pod_status(self, pod_name: str, namespace: str = "default") -> str:
      """Get status of a specific pod"""
      logger.debug(f"Fetching status for pod: {pod_name} in namespace: {namespace}")
        with k8s_error_handling():
            pod = self.v1.read_namespaced_pod(pod_name, namespace)
            return pod.status.phase

    def list_pods(self, namespace: str = "default") -> List[str]:
        """List all pods in a namespace"""
        with k8s_error_handling():
            pods = self.v1.list_namespaced_pod(namespace)
            return [pod.metadata.name for pod in pods.items]

    def get_pod_logs(self, pod_name: str, namespace: str = "default") -> str:
        """Get logs from a specific pod"""
        with k8s_error_handling():
            return self.v1.read_namespaced_pod_log(pod_name, namespace)
    
    def get_pods_by_deployment(self, deployment_name: str, namespace: str = "default") -> List[str]:
        """Get pods created by a specific deployment"""
        with k8s_error_handling():
            deployment = self.apps_v1.read_namespaced_deployment(deployment_name, namespace)
            label_selector = ",".join([f"{k}={v}" for k, v in deployment.spec.selector.match_labels.items()])
            pods = self.v1.list_namespaced_pod(namespace, label_selector=label_selector)
            base_names = [pod.metadata.name.split("-")[0] for pod in pods.items]
            return base_names

    def list_nodes(self) -> List[str]:
        """List all nodes in the cluster"""
        with k8s_error_handling():
            nodes = self.v1.list_node()
            return [node.metadata.name for node in nodes.items]

    def list_services(self, namespace: str = "default") -> List[str]:
        """List all services in a namespace"""
        with k8s_error_handling():
            services = self.v1.list_namespaced_service(namespace)
            return [svc.metadata.name for svc in services.items]

    def list_deployments(self, namespace: str = "default") -> List[str]:
        """List all deployments in a namespace"""
        with k8s_error_handling():
            deployments = self.apps_v1.list_namespaced_deployment(namespace)
            return [deploy.metadata.name for deploy in deployments.items]

    def count_running_pods(self, namespace: str = "default") -> int:
        """Count running pods in a namespace"""
        with k8s_error_handling():
            pods = self.v1.list_namespaced_pod(namespace)
            return len([pod for pod in pods.items if pod.status.phase == "Running"])

    def count_nodes(self) -> int:
        """Count total nodes in the cluster"""
        with k8s_error_handling():
            nodes = self.v1.list_node()
            return len(nodes.items)

    def list_namespaces(self) -> List[str]:
        """List all namespaces"""
        with k8s_error_handling():
            namespaces = self.v1.list_namespace()
            return [ns.metadata.name for ns in namespaces.items]

    def parse_query_with_gpt(self, query: str) -> Tuple[str, Dict[str, Any]]:
        """Use GPT to understand the query and return the appropriate operation and parameters"""
        
        system_message = f"""
        You are a Kubernetes query interpreter. Your task is to analyze natural language queries and map them to specific operations. 
        Available operations and their formats:
        {json.dumps(self.operations, indent=2)}
        
        Return a JSON object with:
        1. "operation": The name of the operation to execute
        2. "parameters": A dictionary of parameters needed for the operation
        
        Example:
        Query: "What is the status of pod 'nginx'?"
        Response: {{"operation": "get_pod_status", "parameters": {{"pod_name": "nginx"}}}}
        """

        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": query}
                ],
                temperature=0
            )
            
            result = json.loads(response.choices[0].message.content)
            return result["operation"], result["parameters"]
            
        except Exception as e:
            logger.error(f"Error parsing query with GPT: {e}")
            raise Exception("Failed to understand the query")

    def process_query(self, query: str) -> str:
        """Process natural language query using GPT for understanding"""
        try:
            
            operation_name, parameters = self.parse_query_with_gpt(query)
            
            
            operation = getattr(self, operation_name)
            
            
            result = operation(**parameters)
            
            
            if isinstance(result, list):
                return ", ".join(result) if result else "No items found"
            elif isinstance(result, int):
                return str(result)
            else:
                return str(result)

        except Exception as e:
            logger.error(f"Error processing query: {e}")
            raise Exception(str(e))


query_processor = KubernetesQueryProcessor()

@app.route('/query', methods=['POST'])
def process_query():
    """
    Process a natural language query about the Kubernetes cluster
    """
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({'error': 'No query provided'}), 400

        answer = query_processor.process_query(data['query'])
        return jsonify({
            'query': data['query'],
            'answer': answer
        })
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000)
