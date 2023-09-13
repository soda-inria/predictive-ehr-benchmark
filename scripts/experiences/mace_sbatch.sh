#!/bin/bash
#SBATCH --job-name=mace
#SBATCH --time 48:00:00
#SBATCH --gres=gpu:t4:0
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --partition gpuT4
#SBATCH --mem=38G
#SBATCH --container-image /scratch/images/sparkhadoop.sqsh  --container-mounts=/export/home/$USER:/export/home/$USER,/export/home/share:/export/home/share,/data/scratch/$USER:/data/scratch/$USER --container-mount-home --container-writable --container-workdir=/
#SBATCH --output=logs/slurm-%j-stdout.log
#SBATCH --error=logs/slurm-%j-stderr.log
export PATH=/usr/hdp/current/accumulo-client/bin:/usr/hdp/current/atlas-server/bin:/usr/hdp/current/beacon-client/bin:/usr/hdp/current/hive-client/bin:/usr/hdp/current/falcon-client/bin:/usr/hdp/current/flume-server/bin:/usr/hdp/current/hadoop-client/bin:/usr/hdp/current/hbase-client/bin:/usr/hdp/current/hadoop-hdfs-client/bin:/usr/hdp/current/hadoop-mapreduce-client/bin:/usr/hdp/current/hadoop-yarn-client/bin:/usr/hdp/current/hive-client/bin:/usr/hdp/current/hive-hcatalog/bin:/usr/hdp/current/hive-server2/bin:/usr/hdp/current/kafka-broker/bin:/usr/hdp/current/mahout-client/bin:/usr/hdp/current/oozie-client/bin:/usr/hdp/current/oozie-server/bin:/usr/hdp/current/phoenix-client/bin:/usr/hdp/current/pig-client/bin:/usr/hdp/share/hst/hst-agent/python-wrap:/usr/hdp/current/slider-client/bin:/usr/hdp/current/sqoop-client/bin:/usr/hdp/current/sqoop-server/bin:/usr/hdp/current/storm-slider-client/bin:/usr/hdp/current/zookeeper-client/bin:/usr/hdp/current/zookeeper-server/bin:/export/home/opt/jupyterhub/conda/bin:/export/home/opt/jupyterhub/node/bin:/export/home/opt/bin:/export/home/cse210037/.user_conda/miniconda/envs/event2vec/bin:/usr/hdp/current/accumulo-client/bin:/usr/hdp/current/atlas-server/bin:/usr/hdp/current/beacon-client/bin:/usr/hdp/current/hive-client/bin:/usr/hdp/current/falcon-client/bin:/usr/hdp/current/flume-server/bin:/usr/hdp/current/hadoop-client/bin:/usr/hdp/current/hbase-client/bin:/usr/hdp/current/hadoop-hdfs-client/bin:/usr/hdp/current/hadoop-mapreduce-client/bin:/usr/hdp/current/hadoop-yarn-client/bin:/usr/hdp/current/hive-client/bin:/usr/hdp/current/hive-hcatalog/bin:/usr/hdp/current/hive-server2/bin:/usr/hdp/current/kafka-broker/bin:/usr/hdp/current/mahout-client/bin:/usr/hdp/current/oozie-client/bin:/usr/hdp/current/oozie-server/bin:/usr/hdp/current/phoenix-client/bin:/usr/hdp/current/pig-client/bin:/usr/hdp/share/hst/hst-agent/python-wrap:/usr/hdp/current/slider-client/bin:/usr/hdp/current/sqoop-client/bin:/usr/hdp/current/sqoop-server/bin:/usr/hdp/current/storm-slider-client/bin:/usr/hdp/current/zookeeper-client/bin:/usr/hdp/current/zookeeper-server/bin:/export/home/opt/jupyterhub/conda/bin:/export/home/opt/jupyterhub/node/bin:/export/home/opt/bin:/usr/hdp/current/accumulo-client/bin:/usr/hdp/current/atlas-server/bin:/usr/hdp/current/beacon-client/bin:/usr/hdp/current/hive-client/bin:/usr/hdp/current/falcon-client/bin:/usr/hdp/current/flume-server/bin:/usr/hdp/current/hadoop-client/bin:/usr/hdp/current/hbase-client/bin:/usr/hdp/current/hadoop-hdfs-client/bin:/usr/hdp/current/hadoop-mapreduce-client/bin:/usr/hdp/current/hadoop-yarn-client/bin:/usr/hdp/current/hive-client/bin:/usr/hdp/current/hive-hcatalog/bin:/usr/hdp/current/hive-server2/bin:/usr/hdp/current/kafka-broker/bin:/usr/hdp/current/mahout-client/bin:/usr/hdp/current/oozie-client/bin:/usr/hdp/current/oozie-server/bin:/usr/hdp/current/phoenix-client/bin:/usr/hdp/current/pig-client/bin:/usr/hdp/share/hst/hst-agent/python-wrap:/usr/hdp/current/slider-client/bin:/usr/hdp/current/sqoop-client/bin:/usr/hdp/current/sqoop-server/bin:/usr/hdp/current/storm-slider-client/bin:/usr/hdp/current/zookeeper-client/bin:/usr/hdp/current/zookeeper-server/bin:/export/home/opt/jupyterhub/conda/bin:/export/home/opt/jupyterhub/node/bin:/export/home/opt/bin:/usr/hdp/current/accumulo-client/bin:/usr/hdp/current/atlas-server/bin:/usr/hdp/current/beacon-client/bin:/usr/hdp/current/hive-client/bin:/usr/hdp/current/falcon-client/bin:/usr/hdp/current/flume-server/bin:/usr/hdp/current/hadoop-client/bin:/usr/hdp/current/hbase-client/bin:/usr/hdp/current/hadoop-hdfs-client/bin:/usr/hdp/current/hadoop-mapreduce-client/bin:/usr/hdp/current/hadoop-yarn-client/bin:/usr/hdp/current/hive-client/bin:/usr/hdp/current/hive-hcatalog/bin:/usr/hdp/current/hive-server2/bin:/usr/hdp/current/kafka-broker/bin:/usr/hdp/current/mahout-client/bin:/usr/hdp/current/oozie-client/bin:/usr/hdp/current/oozie-server/bin:/usr/hdp/current/phoenix-client/bin:/usr/hdp/current/pig-client/bin:/usr/hdp/share/hst/hst-agent/python-wrap:/usr/hdp/current/slider-client/bin:/usr/hdp/current/sqoop-client/bin:/usr/hdp/current/sqoop-server/bin:/usr/hdp/current/storm-slider-client/bin:/usr/hdp/current/zookeeper-client/bin:/usr/hdp/current/zookeeper-server/bin:/export/home/opt/jupyterhub/conda/bin:/export/home/opt/jupyterhub/node/bin:/export/home/opt/bin:/usr/hdp/current/spark-2.4.3-client/bin:/usr/hdp/current/spark-2.4.3-client/bin:/usr/local/hadoop/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/export/home/opt/apps/texlive-20190227/2018/bin/x86_64-linux:/export/home/opt/apps/texlive-20190227/2018/bin/x86_64-linux:/export/home/opt/apps/texlive-20190227/2018/bin/x86_64-linux:/export/home/opt/apps/texlive-20190227/2018/bin/x86_64-linux:/export/home/cse210037/.local/bin:/export/home/cse210037/bin:$PATH
cd '/export/home/cse210037/Matthieu/medical_embeddings_transfer/scripts/experiences'
/export/home/cse210037/.user_conda/miniconda/envs/event2vec-py310/bin/python /export/home/cse210037/Matthieu/medical_embeddings_transfer/medem/experiences/setups/mace_prediction.py 

