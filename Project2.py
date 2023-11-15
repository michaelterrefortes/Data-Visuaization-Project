from dash import Dash, dash_table, dcc, html, Input, Output, callback
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import math
import matplotlib as plt

def Y_a_b(genes, a, b):
  return np.mean(genes[a:b])

def C_a_b(genes, a, b):
  mean = Y_a_b(genes, a, b+1)
  return sum( (np.array(genes[a:b+1]) - mean) ** 2 )

def determine_h(P, i, j, genes):
  N = len(genes)

  if (i == 0 and j > 0):
    return Y_a_b(genes, P[i][j], P[i+1][j]) - Y_a_b(genes, 0, P[i][j]);
  elif (i == j and j > 0):
    return Y_a_b(genes, P[i][j], N) - Y_a_b(genes, P[i-1][j], P[i][j]);
  elif (i == 0 and j == 0):
    return Y_a_b(genes, P[i][j], N) - Y_a_b(genes, 0, P[i][j]);
  else:
    return Y_a_b(genes, P[i][j], P[i+1][j]) - Y_a_b(genes, P[i-1][j], P[i][j]);

def BASC_A(gene):
    gene_og = gene
    gene = np.sort(gene)
    N = len(gene)

    cost_matrix = [[0 for _ in range(N - 1)] for _ in range(N)]
    ind_matrix = [[0 for _ in range(N - 2)] for _ in range(N - 1)]
    P = [[0 for _ in range(N - 2)] for _ in range(N - 2)]

    # Step 1: Compute a Series of Step Function

    # initialization C_i_(0) = c_i_N
    # calculate first cost matrix column with no intermidiate break points
    for i in range(N):
      cost_matrix[i][0] = C_a_b(gene, i, N)

    # Algorithm 1: Calculate optimal step functions
    for j in range(N-2):
      for i in range(N-j-1):
        min_value = math.inf
        min_index = math.inf

        for d in range(N-j-1):
          curr_value = C_a_b(gene, i, d) + cost_matrix[d+1][j]

          if(curr_value < min_value):
            min_value = curr_value
            min_index = d

        cost_matrix[i][j+1] = min_value
        ind_matrix[i][j] = min_index + 1

    #  Algorithm 2: Compute the break points of all optimal step functions
    for j in range(N-2):
      z = j
      P[0][j] = ind_matrix[0][z]
      if(j > 0):
        z = z - 1
        for i in range(1, j+1):
          P[i][j] = ind_matrix[P[i-1][j]][z]
          z = z - 1

    # Step 2: Find Strongest Discontinuity in Each Step Function
    v = [0] * (N-2)

    for j in range(N-2):
      max_value = -math.inf
      max_index = j
      for i in range(j+1):
        h = determine_h(P, i, j, gene)
        z = (gene[P[i][j]] + gene[P[i][j]-1]) / 2
        e = sum( (np.array(gene) - z) ** 2 )
        q_score = h / e
        if(q_score > max_value):
          max_value = q_score
          max_index = i

      v[j] = P[max_index][j]

    # Step 3: Estimate Location and Variation of the Strongest Discontinuities
    thr = (gene[round(np.median(v))-1] + gene[round(np.median(v))]) / 2

    return thr

from sklearn.cluster import KMeans

def K_Means(genes):
    data = np.array(genes).reshape(-1, 1)
    kmeans = KMeans(n_clusters=2)
    kmeans.fit(data)
    c=kmeans.labels_
    genes = np.array(genes)
    groupOne = genes[c==1]
    groupZero = genes[c==0]
    
    thr1 = np.mean(groupOne)
    thr2 = np.mean(groupZero)
    
    thr = (thr1 + thr2) / 2

    return thr

def getSSTOT(x, n, xmean):
    m = 0
    for i in range(n):
        m = m + (x[i] - xmean)**2
    return m


def onestep(x):
    
    n = len(x)
    #step = 0
    xmean = np.mean(x)
    SSTOT = getSSTOT(x, n, xmean)
    
    SSEmin = SSTOT
    
    for i in range(n-1):
        leftMean = np.mean(x[0:i+1])
    
        rightMean = np.mean(x[i+1:n])
        
        SSE = 0
        
        for j in range(n):
            if j < i+1:
                SSE = SSE + (x[j] - leftMean)**2
            else:
                SSE = SSE + (x[j] - rightMean)**2
                    
        
        if SSEmin > SSE:
            SSEmin = SSE
            #print("1:",SSEmin1)
                
            t = (leftMean + rightMean)/2
    
    
    
    return t


def binarize(x):
    
    n = len(x)
    s = np.sort(x)
    d = np.empty(n)
    
    for i in range(n-2):
        d[i] = s[i+1] - s[i]
    
    t = (s[n-1] - s[0])/(n-1)
    
    mn = s[n-1]
    index = 0
    
    for i in range(n-1):
        if d[i] > t and d[i] < mn:
            mn = d[i]
            index = i
            
    z = s[index + 1]
   
    
    return z


df = pd.read_csv('HIVIn(Matlab).csv')

app = Dash(__name__)

app.layout = html.Div([
    dash_table.DataTable(
        id='datatable-interactivity',
        columns=[
            {"name": i, "id": i} for i in df.columns
        ],
        data=df.to_dict('records'),
        column_selectable="single",
        row_selectable="multi",
        selected_columns=[],
        selected_rows=[],
        page_action="native",
        page_current= 0,
        page_size= 10,
    ),
    dcc.Dropdown(
                [{'label': 'Binarize All Genes', 'value':'all'}],
                placeholder="Select to binarize all genes and get thresholds",
                id="dropdown-binarize-all",
                searchable=False),
    html.Div(id='binarize-all'),
    html.Div(id='dropdown-methods'),
    html.Div(id='heatmap-binarize'),
    html.Div(id='select-gene-binarize'),
    html.Div(id='graph-gene-binarize'),
    html.Div(id='select-basc-discontinuity'),
    html.Div(id='graph-gene-discontinuity')
])

@app.callback(
    Output('binarize-all', 'children'),
    Input('dropdown-binarize-all', 'value'))
def binarize_all(all_rows):
    if all_rows is None:
        return 'Select option in order to show all the thresholds by each algorithm of each gene'
    genes = df.values
    rows = df.shape[0]
    
    col_names = {'basc_thr':[], 'kmeans_thr':[], 'onestep_thr':[], 'shmulevich_thr':[]}
    final_df = pd.DataFrame(col_names)
    
    for i in range(rows):
            k_means = K_Means(genes[i])
            basc_a = BASC_A(genes[i])
            one_step = onestep(genes[i])
            shmulevich = binarize(genes[i])
            
            new_row = {'basc_thr':basc_a, 'kmeans_thr':k_means, 'onestep_thr':one_step, 'shmulevich_thr': shmulevich}
            final_df.loc[len(final_df)] = new_row
        
    return dash_table.DataTable(final_df.to_dict('records'), [{"name": i, "id": i} for i in final_df.columns],
                               page_size= 10)
    
            

@app.callback(
    Output('dropdown-methods', 'children'),
    Input('datatable-interactivity', 'selected_rows'))
def display_selected_data(selected_rows):
    if not selected_rows:
        return 'No selected rows'
    return dcc.Dropdown(
                ['All','BASC A', 'K-Means'],
                placeholder="Select binarization method",
                id="dropdown-method",
                searchable=False)

@app.callback(
    Output('heatmap-binarize', 'children'),
    Input('dropdown-method', 'value'),
    Input('datatable-interactivity', 'selected_rows'), prevent_initial_call=True)
def heatmap_binarize(selected_method, selected_rows):
    if selected_rows is None:
        return "No selected rows"
    if selected_method is None:
        return "No method selected"
    
    #selected = df.iloc[selected_rows]
    #gene = selected.values
    #sizeGene = len(gene)
    
    if(selected_method == "K-Means"):
        binarize_vect = []
        labels = []

        for row in selected_rows:
            selected = df.iloc[row]
            gene = selected.values
            sizeGene = len(gene)
            binarize = []
            labels.append("Gene " + str(row))
            thr = K_Means(gene)

            for j in range(sizeGene):
                if(gene[j] <= thr):
                    binarize.append(0) 
                else:
                    binarize.append(1)     

            binarize_vect.append(binarize) 
            
        selected = df.iloc[selected_rows]
        genes = selected.values
        
        data = go.Figure(data=go.Heatmap(
                    z=genes,
                    y = labels,
                    text=binarize_vect,
                    texttemplate="%{text}",
                    textfont={"size":20}))
            
        return dcc.Graph(figure=data)
        
    elif(selected_method == "BASC A"):   
        binarize_vect = []
        labels = []

        for row in selected_rows:
            selected = df.iloc[row]
            gene = selected.values
            sizeGene = len(gene)
            binarize = []
            labels.append("Gene " + str(row))
            thr = BASC_A(gene)

            for j in range(sizeGene):
                if(gene[j] <= thr):
                    binarize.append(0) 
                else:
                    binarize.append(1)     

            binarize_vect.append(binarize) 
            
        selected = df.iloc[selected_rows]
        genes = selected.values
        
        data = go.Figure(data=go.Heatmap(
                    z=genes,
                    y = labels,
                    text=binarize_vect,
                    texttemplate="%{text}",
                    textfont={"size":20}))
            
        return dcc.Graph(figure=data)
        

@app.callback(
    Output('select-gene-binarize', 'children'),
    Input('dropdown-method', 'value'),
    Input('datatable-interactivity', 'selected_rows'), prevent_initial_call=True)
def select_gene_binarize(selected_method, selected_rows):
    if selected_method is None:
        return "No method selected"

    if not selected_rows:
        return "No selected rows"

    return dcc.Dropdown(
        options=[{'label': 'Gene ' + str(row+1), 'value': row} for row in selected_rows],
        placeholder="Select rows",
        id="dropdown-selected-rows")

@app.callback(
    Output('graph-gene-binarize', 'children'),
    Input('dropdown-method', 'value'),
    Input('dropdown-selected-rows', 'value'), prevent_initial_call=True)
def graph_gene_algorithm(selected_method, selected_gene):
    if not selected_gene:
        return 'No specific gene was selected'
    
    selected = df.iloc[selected_gene]
    gene = selected.values
    sizeGene = len(gene)
    
    if(selected_method == 'BASC A'):
        thr = BASC_A(gene)
        data = go.Figure(go.Scatter(x=np.arange(1,sizeGene+1), y=gene, name="Gene "+ str(selected_gene+1)))
        data.add_hline(y=thr, line_width=3, line_dash="dash", line_color="green")
        return dcc.Graph(figure=data)
        
    elif(selected_method == 'K-Means'):
        thr = K_Means(gene)
        data = go.Figure(go.Scatter(x=np.arange(1,sizeGene+1), y=gene, name="Gene "+ str(selected_gene+1)))
        data.add_hline(y=thr, line_width=3, line_dash="dash", line_color="green")
        
        data2 = go.Figure(go.Scatter(x=gene, y=np.ones(sizeGene, dtype=int),mode = 'markers'))
        data2.add_vline(x=thr, line_width=3, line_dash="dash", line_color="green")
        return [dcc.Graph(figure=data), dcc.Graph(figure=data2)]
    
@app.callback(
    Output('select-basc-discontinuity', 'children'),
    Input('dropdown-method', 'value'),
    Input('dropdown-selected-rows', 'value'), prevent_initial_call=True)
def select_basc_discontinuity(selected_method, selected_gene):
    if selected_method is None:
        return "No method selected"

    if not selected_gene:
        return "No selected row for discontinuity"
    
    if(selected_method == 'BASC A'):
    
        selected = df.iloc[selected_gene]
        gene = selected.values
        sizeGene = len(gene)
        
        options = [{'label':'All','value':0}]
        for i in range(sizeGene):
            options.append({'label':'Discontinuity ' + str(i+1), 'value': i+1})
            
        return dcc.Dropdown(
            options=options,
            placeholder="Select discontinuity",
            id="dropdown-selected-discontinuity", value=0)
    
@app.callback(
    Output('graph-gene-discontinuity', 'children'),
    Input('dropdown-selected-rows', 'value'),
    Input('dropdown-selected-discontinuity', 'value'), prevent_initial_call=True)
def graph_discontinuity(selected_gene, selected_discontinuity):
    if selected_discontinuity is None:
        return "No discontinuity selected"

    if not selected_gene:
        return "No selected row for discontinuity"
    
    selected = df.iloc[selected_gene]
    gene = selected.values
    sizeGene = len(gene)
    
    if(selected_discontinuity > 0):
            return 'todavia'
    else:
        x = np.arange(1,sizeGene+1)
        y = gene
        x_dis = []
        y_dis = []
        for i in range(len(x)):
                x_dis.append(i)
                x_dis.append(i+1)
                x_dis.append(None)
                y_dis.append(y[i])
                y_dis.append(y[i])
                y_dis.append(None)
            
        data2 = go.Figure(go.Scatter(x=x_dis, y=y_dis))
        return dcc.Graph(figure=data2)
        

if __name__ == '__main__':
    app.run_server(debug=True)