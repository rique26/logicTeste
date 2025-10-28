# importado conjunto de dados das flores iris
# 150 amostras de flores, 4 atributos e 3 classes (espécies)

# Quais mudanças mínimas é preciso pra que o modelo randonForest 
# preveja uma classe alvo

# Exemplo: 
# instanciou flor
# inicialmente a florestsa classifica-a como setosa
# o que precisa mudar para classificar como versicolor?


# Percorre cada arvore do modelo de decisao randonforest treinado, e retorna 
# os caminhos que levam a classe alvo

def get_target_paths_per_tree(rf: RandomForestClassifier, target_class: int):
    """Retorna lista de caminhos alvo por árvore."""
    per_tree = []
    for est in rf.estimators_:
        per_tree.append(enumerate_target_paths_tree(est, target_class))
    return per_tree  # list of list-of-paths

# prepara o ambiente pra resolver o problema lógico com base nas regras aprendidas pelo modelo 

def setup_solver(rf: RandomForestClassifier, x: np.ndarray): # debug: x = array([4.8, 3.4, 1.6, 0.2])
    """Cria pool, WCNF, thresholds e variáveis y(j,t)."""
    pool = IDPool()
# Ele serve para guardar as regras e restrições que o modelo vai respeitar — tipo:
#“Se variável X for verdadeira, então Y também deve ser”,
# O solver depois lê isso para achar a melhor solução.
    w = WCNF()

    
    thresholds = collect_thresholds([rf]) # debug: {3: [0.6000000014901161, 0.6500000059604645,...]}
    y_vars = {(j, t): pool.id(('y', j, t)) for j, ts in thresholds.items() for t in ts} # debug: {(3, 0.6000000014901161): 1, (3, 0.6500000059604645): 2...}
    return pool, w, thresholds, y_vars

def add_tree_constraints(w: WCNF, pool: IDPool, y_vars: Dict[Tuple[int,float], int],
                         per_tree_paths: List[List[List[Tuple[int,float,str]]]]):
    """Adiciona variáveis z_t, k_{t,p} e todas as constraints de árvores."""
    z_vars = []
    k_vars = {}
    for t_idx, paths in enumerate(per_tree_paths):
        z_t = pool.id(('z', t_idx))
        z_vars.append(z_t)

        if not paths:
            w.append([-z_t])
            continue

        k_list = []
        for p_idx, path in enumerate(paths): 
            k = pool.id(('k', t_idx, p_idx))
            k_vars[(t_idx, p_idx)] = k
            k_list.append(k)
            # k -> conjunção dos testes

            # se um caminho é escolhido, todos os testes desse caminho devem ser satisfeitos pelos atributos da instância

            for feat, thr, d in path: # debug: path = [(3, 0.800000011920929, 'L')] == (x(3) <= 0.8?) 
                lit = y_vars[(feat, thr)]
                w.append([-k, lit] if d == 'R' else [-k, -lit])
            # k -> z_t
            w.append([-k, z_t])
        # z_t -> (∨ k)
        w.append([-z_t] + k_list)
        add_atmost_one(w, k_list)

    return z_vars, k_vars

# adiciona restrições de variáveis need serem verdadeiras
# Calcula quantas árvores precisam concordar para formar a maioria.
def add_majority_constraint(w: WCNF, z_vars: List[int], rf: RandomForestClassifier):
    need = (len(rf.estimators_) // 2) + 1
    pool = IDPool()  # necessário para add_atleast_k
    add_atleast_k(w, z_vars, need, pool)

def extract_solver_result(m, y_vars, thresholds, x, feature_names, k_vars, per_tree_paths):
    """Decodifica resultado do solver: custo, mudanças legíveis e caminhos escolhidos."""
    if m is None:
        return None, [], {}
    cost, changes = diff_cost_from_model(m, y_vars, thresholds, x)
    changes_fmt = fmt_changes(changes, feature_names)
    pos = set(l for l in m if l > 0)
    chosen_paths = {}
    for (t_idx, p_idx), kv in k_vars.items():
        if kv in pos:
            chosen_paths.setdefault(t_idx, []).append(per_tree_paths[t_idx][p_idx])
    return cost, changes_fmt, chosen_paths

# Encontrar o menor número de mudanças necessárias na instância x para que a floresta rf vote na classe target_class.

def solve_forest_min_changes(rf: RandomForestClassifier, x: np.ndarray, target_class: int,
                                     feature_names: List[str]) -> Tuple[int, List[str], Dict[int, List[Tuple[int,float,str]]]]:

    per_tree_paths = get_target_paths_per_tree(rf, target_class)
    pool, w, thresholds, y_vars = setup_solver(rf, x)
    add_sigma_monotonicity(w, thresholds, y_vars)
    add_soft_tx(w, x, thresholds, y_vars)

# Traduz as regras das árvores em restrições lógicas.
# Cada árvore vira um conjunto de proposições
# estrutura de fórmulas lógicas 
    z_vars, k_vars = add_tree_constraints(w, pool, y_vars, per_tree_paths)

                                         
    add_majority_constraint(w, z_vars, rf)

    with RC2(w) as rc2:
        m = rc2.compute()

    return extract_solver_result(m, y_vars, thresholds, x, feature_names, k_vars, per_tree_paths)

















