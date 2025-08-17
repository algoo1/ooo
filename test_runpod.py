#!/usr/bin/env python3
"""
Test script for CLIP Embedding API on RunPod
"""

import requests
import json
import time
from typing import Dict, Any, Optional

class RunPodTester:
    def __init__(self, endpoint_id: str, api_key: str):
        self.endpoint_id = endpoint_id
        self.api_key = api_key
        self.base_url = "https://api.runpod.ai/v2"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def run_sync(self, input_data: Dict[str, Any], timeout: int = 300) -> Dict[str, Any]:
        """Run synchronous request"""
        url = f"{self.base_url}/{self.endpoint_id}/runsync"
        
        payload = {"input": input_data}
        
        print(f"🚀 Sending sync request to: {url}")
        print(f"📤 Payload: {json.dumps(payload, indent=2)}")
        
        try:
            response = requests.post(
                url, 
                json=payload, 
                headers=self.headers, 
                timeout=timeout
            )
            
            print(f"📊 Status Code: {response.status_code}")
            print(f"📋 Response Headers: {dict(response.headers)}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Success: {json.dumps(result, indent=2)}")
                return result
            else:
                error_text = response.text
                print(f"❌ Error Response: {error_text}")
                return {
                    "error": f"HTTP {response.status_code}: {error_text}",
                    "status_code": response.status_code
                }
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Request Exception: {str(e)}")
            return {"error": f"Request failed: {str(e)}"}
    
    def run_async(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run asynchronous request"""
        # Submit job
        url = f"{self.base_url}/{self.endpoint_id}/run"
        payload = {"input": input_data}
        
        print(f"🚀 Submitting async job to: {url}")
        print(f"📤 Payload: {json.dumps(payload, indent=2)}")
        
        try:
            response = requests.post(url, json=payload, headers=self.headers)
            
            if response.status_code != 200:
                return {
                    "error": f"HTTP {response.status_code}: {response.text}",
                    "status_code": response.status_code
                }
            
            job_data = response.json()
            job_id = job_data.get("id")
            
            if not job_id:
                return {"error": "No job ID returned", "response": job_data}
            
            print(f"✅ Job submitted: {job_id}")
            
            # Poll for results
            status_url = f"{self.base_url}/{self.endpoint_id}/status/{job_id}"
            
            for attempt in range(60):  # 5 minutes max
                time.sleep(5)
                status_response = requests.get(status_url, headers=self.headers)
                
                if status_response.status_code == 200:
                    status_data = status_response.json()
                    status = status_data.get("status")
                    
                    print(f"📊 Status check {attempt + 1}: {status}")
                    
                    if status == "COMPLETED":
                        output = status_data.get("output", {})
                        print(f"✅ Job completed: {json.dumps(output, indent=2)}")
                        return output
                    elif status in ["FAILED", "CANCELLED", "TIMED_OUT"]:
                        error = status_data.get("error", "Unknown error")
                        print(f"❌ Job failed: {error}")
                        return {"error": f"Job {status}: {error}"}
                else:
                    print(f"❌ Status check failed: {status_response.status_code}")
            
            return {"error": "Job timed out"}
            
        except requests.exceptions.RequestException as e:
            return {"error": f"Request failed: {str(e)}"}

def test_cases():
    """Define test cases"""
    return [
        {
            "name": "Text Only",
            "input": {
                "text": "A beautiful sunset over the ocean"
            }
        },
        {
            "name": "Image Only", 
            "input": {
                "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png"
            }
        },
        {
            "name": "Both Text and Image",
            "input": {
                "text": "A red car on the road",
                "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/9/9a/Gull_portrait_ca_usa.jpg/300px-Gull_portrait_ca_usa.jpg"
            }
        },
        {
            "name": "Invalid Input (should fail gracefully)",
            "input": {
                "invalid_field": "test"
            }
        },
        {
            "name": "Empty Input (should fail gracefully)",
            "input": {}
        }
    ]

def main():
    """Main testing function"""
    # Configuration - REPLACE THESE WITH YOUR VALUES
    ENDPOINT_ID = "YOUR_ENDPOINT_ID_HERE"  # Replace with your RunPod endpoint ID
    API_KEY = "YOUR_API_KEY_HERE"          # Replace with your RunPod API key
    
    if ENDPOINT_ID == "YOUR_ENDPOINT_ID_HERE" or API_KEY == "YOUR_API_KEY_HERE":
        print("❌ Please update ENDPOINT_ID and API_KEY in the script")
        return
    
    tester = RunPodTester(ENDPOINT_ID, API_KEY)
    
    print("=" * 60)
    print("🧪 CLIP Embedding API Test Suite")
    print("=" * 60)
    
    for i, test_case in enumerate(test_cases(), 1):
        print(f"\n🔬 Test {i}: {test_case['name']}")
        print("-" * 40)
        
        # Test sync endpoint
        print(f"\n📡 Testing SYNC endpoint...")
        sync_result = tester.run_sync(test_case["input"])
        
        # Only test async if sync worked
        if "error" not in sync_result:
            print(f"\n📡 Testing ASYNC endpoint...")
            async_result = tester.run_async(test_case["input"])
        
        print("\n" + "=" * 60)
    
    print("🎯 Testing completed!")

if __name__ == "__main__":
    main()
