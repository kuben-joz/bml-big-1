start = """name: mpi-cluster

x-common-service: &common-service-template
  build:
    context: .
    dockerfile: Dockerfile
  networks:
    - my-net
  volumes:
    - type: bind
      source: /home/jakub/Code/BML/bml-big-1
      target: /workspace
      #read_only: true

services:
"""

end = """
networks:
  my-net:
"""
num_cont = 16
with open("compose1.yaml", "w") as file:
    file.write(start)
    for i in range(1, num_cont + 1):
        service = f'''  
    container{i}:
      <<: *common-service-template
      container_name: container{i}
      ports:
        - "{i}022:22"
      {'command: /bin/bash -c "/setup.sh"' if i == 1 else ""}
    
    '''
        file.write(service)
    file.write(end)


with open("../mpi/hosts", "w") as file:
    for i in range(1, num_cont + 1):
        file.write(f"container{i} max_slots=1\n")
