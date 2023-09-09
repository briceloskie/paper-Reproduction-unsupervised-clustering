import networkx as nx
from sklearn.datasets import load_iris
from sklearn.metrics import euclidean_distances, adjusted_rand_score, rand_score
def single_affinty(similarity_matrix,k):
    #第一次迭代循环
    G,result=initial_iteration(similarity_matrix)
    while(True):
        G,result,edges=iteration_loop(similarity_matrix,result,G)
        clusters=nx.number_connected_components(G)
        if clusters<=k:
            break
    if clusters<k:
        result=triming_edges(G,edges,similarity_matrix,k)
    return result
#将最后一次的连边进行排序，然后从大到小进行削边
def triming_edges(G,edges,similarity_matrix,k):
    #此操作是删除重复的边
    for i in range(len(edges)-1):
        for j in range(len(edges)-1-i):
            if similarity_matrix[edges[j][0],edges[j][1]]<similarity_matrix[edges[j+1][0],edges[j+1][1]]:
                edges[j],edges[j+1]=edges[j+1],edges[j]
    for i in edges:
        #如果有边的话就进行切除(因为连边会有两个边同时是最近邻的情况)
        if G.has_edge(i[0],i[1])==True:
            G.remove_edge(i[0],i[1])
        if nx.number_connected_components(G)==k:
            break
    #输出最后的结果
    result=[]
    S = [G.subgraph(c).copy() for c in nx.connected_components(G)]
    for i in S:
        result.append(list(i.nodes))
    return result
#循环迭代,知道簇的个数变为小于等于k的个数
def iteration_loop(similarity_matrix,result,G):
    #找出每个簇的候选人
    edges=[]
    for cluster1 in result:
        # 找出每个簇的候选人
        candidate_vec = []
        for cluster2 in result:
            if cluster2!=cluster1:
                edges_to_add=single_linage_cal(cluster1,cluster2,similarity_matrix)
                candidate_vec.append(edges_to_add)
        #找到后选边的最小边
        edge=candidate_cal(candidate_vec,similarity_matrix)
        edges.append(edge)
    #新增边
    G.add_edges_from(edges)
    #输出迭代的结果
    result=[]
    S = [G.subgraph(c).copy() for c in nx.connected_components(G)]
    for i in S:
        result.append(list(i.nodes))
    return G,result,edges
def candidate_cal(candidate_vec,similarity_matrix):
    minimum_index=candidate_vec[0]
    minimum_value=similarity_matrix[(minimum_index[0],minimum_index[1])]
    for edge in candidate_vec:
        if similarity_matrix[edge[0],edge[1]]<=minimum_value:
            minimum_value=similarity_matrix[edge[0],edge[1]]
            minimum_index=edge
    return minimum_index
#找到两个子簇的single-linkage
def single_linage_cal(cluster1,cluster2,similarity_matrix):
    minimum_index=[cluster1[0],cluster2[0]]
    minimum_value=similarity_matrix[cluster1[0],cluster2[0]]
    for i in cluster1:
        for j in cluster2:
            if similarity_matrix[i,j]<minimum_value:
                minimum_index=(i,j)
    return minimum_index
#初始化迭代
def initial_iteration(similarity_matrix):
    G=nx.Graph()
    length=len(similarity_matrix)
    neighbor_vec=[]
    for i in range(length):
        if i==0:
            minimum_value = similarity_matrix[1, 0]
            min_index=[1,0]
        else:
            minimum_value = similarity_matrix[i, 0]
            min_index = (i, 0)
        for j in range(length):
            if similarity_matrix[i,j]<minimum_value and i!=j:
                minimum_value=similarity_matrix[i,j]
                min_index=(i,j)
        neighbor_vec.append(min_index)
    G.add_edges_from(neighbor_vec)
    #输出第一次迭代的结果
    result=[]
    S = [G.subgraph(c).copy() for c in nx.connected_components(G)]
    for i in S:
        result.append(list(i.nodes))
    return G,result
def result_to_predict_vec(result):
    tranversal_dict={}
    predict_vec=[]
    for i in range(len(result)):
        for j in result[i]:
            tranversal_dict[j]=i
    for i in range(len(tranversal_dict)):
        predict_vec.append(tranversal_dict[i])
    return predict_vec

if __name__ == '__main__':
    iris = load_iris()
    data = iris.data  # 特征数据
    real_labels = iris.target  # 目标数据
    similarity_matrix = euclidean_distances(data)
    result = single_affinty(similarity_matrix, k=3)
    predicted_labels = result_to_predict_vec(result)
    ari_score = rand_score(real_labels, predicted_labels)
    print("ARI指标:", ari_score)





