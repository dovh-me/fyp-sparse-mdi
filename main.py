import node.node as node
import asyncio
import sys

if __name__ == '__main__':
    coordinator_ip=sys.argv[1] if len(sys.argv) > 1 else None

    if coordinator_ip is None:
        asyncio.run(node.main()) 
    else:
        asyncio.run(node.main(coordinator_ip=coordinator_ip)) 
    
