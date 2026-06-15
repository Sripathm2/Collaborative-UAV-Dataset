#### Start of Makefile ####
########################################################################
#### a. Setting up and debugging workspace commands
########################################################################

home_dir := /users/something/
root_dir := /mydata/local/

######## a.0 Setting up containernet (Native installation) ######

install-containernet-and-requirements-part1: ## not tested
# 	sudo sed -i 's|http://us.archive.ubuntu.com|https://us.archive.ubuntu.com|g' /etc/apt/sources.list
	sudo apt-get update
	sudo apt-get install -y ansible python3.10-venv tshark parallel htop
	cd $(home_dir) && git clone https://github.com/containernet/containernet.git
	cd $(home_dir) && sudo ansible-playbook -i "localhost," -c local containernet/ansible/install.yml
	cd $(home_dir) && python3 -m venv venv
	sudo su
	sudo docker build --no-cache --tag=uav_nodes -f Docker/Dockerfile.node Docker/


install-containernet-and-requirements-part2:
	(cd $(home_dir) && \
		source venv/bin/activate && \
		pip3 install wheel && \
		cd containernet && pip3 install .)
	(cd $(root_dir) && \
		source $(home_dir)/venv/bin/activate && \
		pip3 install -r requirements.txt)
	unzip UAV_data.zip
	rm -rf __MACOSX UAV_data.zip
	mv UAV_data ./Network-simulator/
	cp ./Cloudlab-utilities/* ./	
	sudo su
	tmux new-session -d -s mysession 'source /users/mishra60/venv/bin/activate && ./start_collection.sh'

######## a.3 cleaning up ########

clean:
	-sudo docker stop $$(sudo docker ps -a -q --filter="name=mn.*")
	-sudo docker rm $$(sudo docker ps -a -q --filter="name=mn.*")
	-mn -c
	-rm -rf *.png
