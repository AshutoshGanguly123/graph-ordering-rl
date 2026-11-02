"""Read-only compute-node/GPU guard. Standard library only."""
import os
import re
import subprocess
import time

def inspect():
    node=os.uname().nodename.split('.')[0]
    if not re.fullmatch(r'gpu-\d+',node):raise RuntimeError('Compute node required')
    reservation=subprocess.check_output(['scontrol','--details','--oneliner','show','node',node],text=True)
    if re.search(r'\bState=([^ ]+)',reservation).group(1) not in ['IDLE','MIXED']:
        raise RuntimeError('Ineligible node state')
    match=re.search(r'\bGresUsed=gpu:[^(): ]+:(\d+)\(IDX:([^)]*)\)',reservation)
    if not match:raise RuntimeError('Ambiguous reservations')
    count,indices=int(match.group(1)),match.group(2);reserved=set()
    if count:
        for item in indices.split(','):
            if re.fullmatch(r'\d+',item):reserved.add(int(item))
            elif re.fullmatch(r'\d+-\d+',item):
                lo,hi=map(int,item.split('-'))
                if lo>hi:raise RuntimeError('Bad reservation range')
                reserved.update(range(lo,hi+1))
            else:raise RuntimeError('Ambiguous reservation indices')
        if len(reserved)!=count:raise RuntimeError('Reservation count mismatch')
    elif indices!='N/A':raise RuntimeError('Ambiguous empty reservation')
    inventory=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,uuid,memory.total,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name','--format=csv,noheader'],text=True)
    cards=[]
    for line in inventory.strip().splitlines():
        index,name,uuid,total,used,util=[s.strip() for s in line.split(',')]
        if int(index) not in reserved and int(total)>=30000 and int(used)==0 and int(util)==0 and uuid not in apps:
            cards.append(dict(index=int(index),uuid=uuid,name=name,memory_mib=int(total)))
    if len(cards)<4:raise RuntimeError('Four physically empty, unreserved >=30GB cards required')
    return dict(node=node,reservations=reservation,inventory=inventory,processes=apps,selected=cards[:4],time=time.time())
