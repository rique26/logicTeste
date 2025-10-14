# ----------------------------- FLORESTA: maioria via disjunção de caminhos -----------------------------

** Essas funções fazem parte de um sistema que gera explicações contrastivas para modelos de aprendizado de máquina baseados em árvores de decisão
**Em resumo, o objetivo é responder perguntas como:
“O que precisa mudar no exemplo x para que a floresta mude sua decisão para a classe C?”


//Essa funcao coleta todos os caminhos das árvores dentro da floresta (RandomForestClassifier) que terminam na classe desejada (target_class).

def enumerate_target_paths_forest(rf: RandomForestClassifier, target_class: int): # Define função que coleta caminhos das folhas de uma floresta para uma classe alvo
    """Retorna lista por árvore: paths_t = [ [(feat,thr,dir), ...], ... ] apenas para folhas da classe alvo"""
    per_tree = []  # Lista que armazenará os caminhos de cada árvore
    for est in rf.estimators_: # Percorre cada árvore (estimator) da floresta
        per_tree.append(enumerate_target_paths_tree(est, target_class)) # Coleta caminhos da árvore que levam à classe alvo
    return per_tree  # list of list-of-paths / Retorna lista de listas de caminhos (um por árvore)






//Essa é a função principal: ela usa as informações da floresta (os caminhos) para descobrir o menor número de mudanças em x que fariam a floresta prever a classe desejada.

def solve_forest_min_changes(rf: RandomForestClassifier, x: np.ndarray, target_class: int, # Define função que encontra o menor conjunto de mudanças para um RandomForest
                             feature_names: List[str]) -> Tuple[int, List[str], Dict[int, List[Tuple[int,float,str]]]]: # Retorna (custo, mudanças formatadas, caminhos escolhidos por árvore)
    """
    Retorna (custo, mudanças_fmt, caminhos_escolhidos_por_árvore)
    f: para cada árvore t, escolhe no máximo 1 caminho alvo (variáveis k_{t,p}),
       z_t é verdadeiro sse algum k_{t,p} é verdadeiro; maioria em z_t.
    """


//Coleta os caminhos da floresta que levam à classe desejada.
    per_tree_paths = enumerate_target_paths_forest(rf, target_class) 
    # se poucas árvores têm caminho-alvo, maioria pode ser impossível; deixamos o solver decidir (UNSAT). Observação: solver decide se não há maioria possível
    pool = IDPool() # Cria gerenciador de IDs de variáveis booleanas
    w = WCNF()  # Cria fórmula CNF ponderada vazia onde serão armazenadas as restrições.


//Criação das variáveis
    thresholds = collect_thresholds([rf]) 
    y = {(j,t): pool.id(('y', j, t)) for j, ts in thresholds.items() for t in ts} # Cada variável y(j,t) representa uma condição binária
//Assim, todos os nós de decisão das árvores viram variáveis booleanas.


//Restrições gerais sobre as variáveis
    add_sigma_monotonicity(w, thresholds, y) # Adiciona restrições de monotonicidade nas variáveis
//Ou seja, se y(j,10) é verdadeiro, então y(j,5) também deve ser.

    add_soft_tx(w, x, thresholds, y) # Adiciona cláusulas suaves conforme o vetor x original


//Criar variáveis z_t e k_{t,p}
    z_vars = [] # Lista de variáveis z_t, uma por árvore (indica se caminho alvo foi escolhido), (ou seja, se tem algum caminho ativo na árvore que leva à classe desejada).
    k_vars = {}  # representa o caminho escolhido




// Modelar cada árvore
//Transformar cada árvore de decisão em um conjunto de variáveis booleanas e restrições lógicas.. até aqui. para que o resolvedor MaxSAT determine as mudanças mínimas em x para alcançar a classe alvo.


	// pra cada árvore cria uma variável z_t
    for t_idx, paths in enumerate(per_tree_paths): # Percorre todas as árvores e seus caminhos
        z_t = pool.id(('z', t_idx)) 
        z_vars.append(z_t) # Adiciona z_t à lista global
        if not paths:   # Se a árvore não possui caminho que leva à classe alvo
            # nenhuma folha-alvo nesta árvore: força ¬z_t
            w.append([-z_t]) # Adiciona cláusula forçando z_t a ser falso
            continue # Passa para a próxima árvore

        k_list = [] # Lista local de variáveis k_{t,p} para os caminhos da árvore
        for p_idx, path in enumerate(paths):  # Percorre cada caminho possível dentro da árvore
            k = pool.id(('k', t_idx, p_idx))  # Cria variável k_{t,p}, indicando se o caminho p foi escolhido
            k_vars[(t_idx, p_idx)] = k   # Salva a variável no dicionário
            k_list.append(k)  # Adiciona à lista local
            # k -> (conjunção dos testes do caminho)
            for (feat, thr, d) in path:  # Para cada teste do caminho (feature, threshold, direção)
                lit = y[(feat, thr)] # Obtém a variável y correspondente
                w.append([-k,  lit] if d=='R' else [-k, -lit]) # Adiciona cláusula: se k é verdadeiro, o teste deve ser satisfeito
            # k -> z_t
            w.append([-k, z_t]) # Adiciona cláusula: se algum k é verdadeiro, então z_t também é

        # z_t -> (∨ k)
        w.append([-z_t] + k_list) # Se z_t é verdadeiro, ao menos um dos k deve ser verdadeiro
        # (opcional, mas ajuda) no máximo um caminho escolhido por árvore
        add_atmost_one(w, k_list)  # Impõe que no máximo um caminho pode ser selecionado por árvore








// aqui ele prever a classe alvo para que a floresta mude a previsão.
//Exemplo: se temos 5 árvores, pelo menos 3 devem 
    need = (len(rf.estimators_) // 2) + 1 # Define número mínimo de árvores necessárias para maioria (maioria simples)
    add_atleast_k(w, z_vars, need, pool)  # Adiciona restrição: pelo menos "need" árvores devem prever a classe alvo


    with RC2(w) as rc2: # Inicializa resolvedor MaxSAT RC2 com a fórmula construída
        m = rc2.compute() # Computa a solução ótima (modelo com custo mínimo)
    if m is None: # Se não há solução satisfatória
        return None, [], {}  # Retorna valores vazios





// Decodificar o resultado
cost = custo total das mudanças.
changes_fmt = formato legível (ex: “idade de 25 → 35”, “salário de 1200 → 2500”).

    # custo e mudanças
    cost, changes = diff_cost_from_model(m, y, thresholds, x) # Calcula o número de variáveis y diferentes do vetor x original
    changes_fmt = fmt_changes(changes, feature_names) # Formata as diferenças em texto legível




//Ver quais caminhos foram escolhidos

    # decodificar caminhos escolhidos
    pos = set(l for l in m if l > 0) # Conjunto das variáveis verdadeiras no modelo
    chosen_paths = {}  # Dicionário para armazenar os caminhos escolhidos por árvore
    for (t_idx, p_idx), kv in k_vars.items(): # Percorre todas as variáveis k_{t,p}
        if kv in pos:  # Se o caminho foi escolhido (k verdadeiro)
            chosen_paths.setdefault(t_idx, []).append(per_tree_paths[t_idx][p_idx]) # Adiciona o caminho correspondente à árvore t





//Retornar tudo

// Saída final:
//cost → número de alterações necessárias.
//changes_fmt → explicações legíveis (“o que mudar em x”).
//chosen_paths → quais caminhos nas árvores levaram à classe alvo.


    return cost, changes_fmt, chosen_paths # Retorna custo total, mudanças formatadas e caminhos escolhidos
