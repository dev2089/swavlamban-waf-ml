"""
Demo Data Generator for WAF ML System
Generates realistic threat data for testing and demonstration
"""

import asyncio
import aiohttp
import random
import time
from datetime import datetime
from typing import List, Dict

# Sample attack patterns
ATTACK_PATTERNS = {
    'SQL_INJECTION': [
        "' OR '1'='1",
        "'; DROP TABLE users--",
        "' UNION SELECT * FROM passwords--",
        "admin'--",
        "1' OR 1=1--",
    ],
    'XSS': [
        "<script>alert('XSS')</script>",
        "<img src=x onerror=alert('XSS')>",
        "javascript:alert('XSS')",
        "<iframe src='javascript:alert(1)'>",
        "<body onload=alert('XSS')>",
    ],
    'PATH_TRAVERSAL': [
        "../../../etc/passwd",
        "..\\..\\..\\windows\\system32",
        "....//....//....//etc/passwd",
        "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd",
    ],
    'COMMAND_INJECTION': [
        "; ls -la",
        "| cat /etc/passwd",
        "&& whoami",
        "`cat /etc/shadow`",
    ]
}

# Sample IPs
ATTACKER_IPS = [
    "192.168.1.100",
    "10.0.0.50",
    "172.16.0.200",
    "203.0.113.45",
    "198.51.100.23",
]

# Sample endpoints
ENDPOINTS = [
    "/api/login",
    "/api/users",
    "/admin/dashboard",
    "/api/search",
    "/uploads/files",
    "/api/products",
    "/user/profile",
]

class DemoDataGenerator:
    def __init__(self, api_url: str = "http://localhost:8000"):
        self.api_url = api_url
        self.running = False

    async def generate_threat(self) -> Dict:
        """Generate a random threat request"""
        threat_type = random.choice(list(ATTACK_PATTERNS.keys()))
        pattern = random.choice(ATTACK_PATTERNS[threat_type])

        request_data = {
            "method": random.choice(["GET", "POST", "PUT", "DELETE"]),
            "uri": random.choice(ENDPOINTS) + f"?input={pattern}",
            "source_ip": random.choice(ATTACKER_IPS),
            "headers": {
                "User-Agent": "Mozilla/5.0 (Attacker)",
                "Content-Type": "application/json"
            },
            "query_params": {
                "input": pattern
            },
            "body": pattern if random.random() > 0.5 else None,
            "user_agent": "Mozilla/5.0 (Attacker)"
        }

        return request_data

    async def generate_normal_request(self) -> Dict:
        """Generate a normal request"""
        request_data = {
            "method": random.choice(["GET", "POST"]),
            "uri": random.choice(ENDPOINTS),
            "source_ip": f"192.168.{random.randint(1, 255)}.{random.randint(1, 255)}",
            "headers": {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
                "Content-Type": "application/json"
            },
            "query_params": {
                "id": str(random.randint(1, 1000))
            },
            "body": None,
            "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
        }

        return request_data

    async def send_request(self, request_data: Dict):
        """Send request to API for analysis"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.api_url}/api/analyze",
                    json=request_data,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    result = await response.json()
                    threat_status = "THREAT" if result.get('threat_detected') else "NORMAL"
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] {threat_status} - "
                          f"{request_data['method']} {request_data['uri'][:50]} - "
                          f"Confidence: {result.get('confidence', 0):.2f}")
                    return result
        except Exception as e:
            print(f"Error sending request: {e}")
            return None

    async def generate_continuous_traffic(self, requests_per_minute: int = 30, threat_ratio: float = 0.3):
        """Generate continuous traffic with both normal and malicious requests"""
        print(f"Starting traffic generation: {requests_per_minute} req/min, {threat_ratio*100}% threats")
        print("=" * 80)

        self.running = True
        interval = 60 / requests_per_minute

        while self.running:
            try:
                # Decide if this should be a threat or normal request
                if random.random() < threat_ratio:
                    request_data = await self.generate_threat()
                else:
                    request_data = await self.generate_normal_request()

                # Send request
                await self.send_request(request_data)

                # Wait before next request
                await asyncio.sleep(interval)

            except KeyboardInterrupt:
                print("\nStopping traffic generation...")
                self.running = False
                break
            except Exception as e:
                print(f"Error in traffic generation: {e}")
                await asyncio.sleep(1)

    async def generate_burst(self, count: int = 100, threat_ratio: float = 0.5):
        """Generate a burst of requests"""
        print(f"Generating burst of {count} requests ({threat_ratio*100}% threats)")
        print("=" * 80)

        tasks = []
        for i in range(count):
            if random.random() < threat_ratio:
                request_data = await self.generate_threat()
            else:
                request_data = await self.generate_normal_request()

            tasks.append(self.send_request(request_data))

            # Add small delay every 10 requests
            if (i + 1) % 10 == 0:
                await asyncio.sleep(0.5)

        # Wait for all requests to complete
        results = await asyncio.gather(*tasks)

        # Print summary
        threats_detected = sum(1 for r in results if r and r.get('threat_detected'))
        print(f"\nBurst complete: {threats_detected}/{count} threats detected")

    async def generate_targeted_attack(self, attack_type: str, duration_seconds: int = 60):
        """Generate a targeted attack simulation"""
        if attack_type not in ATTACK_PATTERNS:
            print(f"Unknown attack type: {attack_type}")
            return

        print(f"Simulating {attack_type} attack for {duration_seconds} seconds")
        print("=" * 80)

        start_time = time.time()
        count = 0

        while time.time() - start_time < duration_seconds:
            pattern = random.choice(ATTACK_PATTERNS[attack_type])
            request_data = {
                "method": random.choice(["GET", "POST"]),
                "uri": random.choice(ENDPOINTS) + f"?input={pattern}",
                "source_ip": random.choice(ATTACKER_IPS[:2]),  # Use same IPs
                "headers": {
                    "User-Agent": "Mozilla/5.0 (Attacker)",
                    "Content-Type": "application/json"
                },
                "query_params": {"input": pattern},
                "body": pattern,
                "user_agent": "Mozilla/5.0 (Attacker)"
            }

            await self.send_request(request_data)
            count += 1
            await asyncio.sleep(0.5)

        print(f"\nAttack simulation complete: {count} requests sent")

async def main():
    """Main demo function"""
    generator = DemoDataGenerator()

    print("WAF ML Demo Data Generator")
    print("=" * 80)
    print("\nOptions:")
    print("1. Continuous traffic (press Ctrl+C to stop)")
    print("2. Generate burst of requests")
    print("3. Simulate targeted attack")
    print("4. Quick demo (10 requests)")

    try:
        choice = input("\nSelect option (1-4): ").strip()

        if choice == "1":
            rpm = int(input("Requests per minute (default 30): ") or "30")
            threat_ratio = float(input("Threat ratio 0-1 (default 0.3): ") or "0.3")
            await generator.generate_continuous_traffic(rpm, threat_ratio)

        elif choice == "2":
            count = int(input("Number of requests (default 100): ") or "100")
            threat_ratio = float(input("Threat ratio 0-1 (default 0.5): ") or "0.5")
            await generator.generate_burst(count, threat_ratio)

        elif choice == "3":
            print("\nAttack types:", ", ".join(ATTACK_PATTERNS.keys()))
            attack_type = input("Attack type: ").strip().upper()
            duration = int(input("Duration in seconds (default 60): ") or "60")
            await generator.generate_targeted_attack(attack_type, duration)

        elif choice == "4":
            await generator.generate_burst(10, 0.5)

        else:
            print("Invalid option")

    except KeyboardInterrupt:
        print("\n\nDemo stopped by user")
    except Exception as e:
        print(f"\nError: {e}")

if __name__ == "__main__":
    asyncio.run(main())
