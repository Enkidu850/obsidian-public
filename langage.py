import re

# Table des variables
variables = {}

# Tokenizer
token_specification = [
    ('INT',           r'INT'),
    ('FLOAT',         r'FLOAT'),
    ('STR',           r'STR'),
    ('PRINT',         r'PRINT'),
    ('BOOL',          r'BOOL'),
    ('POINT',         r'POINT'),
    ('DISTANCE',      r'DISTANCE'),
    ('LINE',          r'LINE'),
    ('POLYGON',       r'POLYGON'),
    ('MAP',           r'MAP'),
    ('BOOL_TRUE',     r'\bTRUE\b'),
    ('BOOL_FALSE',    r'\bFALSE\b'),
    ('EPSG',          r'\[\d+\]'),
    ('LIST_NAME',     r'[a-zA-Z_][a-zA-Z0-9_]*\[\]'),
    ('IDENT',         r'[a-zA-Z_][a-zA-Z0-9_]*#'),
    ('SPATIAL_IDENT', r'[a-zA-Z_][a-zA-Z0-9_]*\$'),
    ('LPAREN',        r'\('),
    ('RPAREN',        r'\)'),
    ('ASSIGN',        r'<-'),
    ('REAL',          r'\d+\.\d+'), # FLOAT
    ('NUMBER',        r'\d+'), # INTEGER
    ('LIST',          r'LIST'),
    ('LBRACK',        r'\['),
    ('RBRACK',        r'\]'),
    ('LBRACE',        r'\{'),
    ('RBRACE',        r'\}'),
    ('COMMA',         r','),
    ('SEMICOLON',     r';'),
    ('SKIP',          r'[ \t]+'),
    ('NEWLINE',       r'\n'),
    ('PLUS',          r'\+'),
    ('MINUS',         r'-'),
    ('MULT',          r'\*'),
    ('DIV',           r'/'),
    ('CHAR',          r'"[^"\n]*"'), # STRING
    ('MISMATCH',      r'.'),  # tout le reste
]

tok_regex = '|'.join('(?P<%s>%s)' % pair for pair in token_specification)
get_token = re.compile(tok_regex).match

class Point:
    def __init__(self, x, y, epsg=4326):
        self.x = x
        self.y = y
        self.epsg = epsg

    def __repr__(self):
        return f"({self.x}, {self.y}) [EPSG:{self.epsg}]"

    def distance_to(self, other):
        if self.epsg != other.epsg:
            raise ValueError("Les points doivent être dans le même système de coordonnées (EPSG).")
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5

class Line:
    def __init__(self, points, epsg=4326):
        if len(points) < 2:
            raise ValueError("Une LINE doit contenir au moins deux points.")
        self.points = points
        self.epsg = epsg

    def __repr__(self):
        return f"({', '.join(str(p) for p in self.points)})"

    def length(self):
        return sum(self.points[i].distance_to(self.points[i+1]) for i in range(len(self.points) - 1))

class Polygon:
    def __init__(self, points, epsg=4326):
        if len(points) < 3:
            raise ValueError("Un POLYGON doit contenir au moins trois sommets.")
        self.points = points
        if points[0] != points[-1]:
            self.points.append(points[0])  # fermeture automatique
        self.epsg = epsg

    def __repr__(self):
        return f"({', '.join(str(p) for p in self.points)})"

    def perimeter(self):
        return sum(self.points[i].distance_to(self.points[i+1]) for i in range(len(self.points) - 1))

def tokenize(code):
    pos = 0
    mo = get_token(code, pos)
    while mo:
        kind = mo.lastgroup
        value = mo.group()
        if kind not in ('SKIP', 'NEWLINE'):
            #print(f"[DEBUG TOKEN] {kind}: {value}")  
            yield (kind, value)
        pos = mo.end()
        mo = get_token(code, pos)
    if pos != len(code):
        raise RuntimeError(f'Unexpected character {code[pos]}')

def parse_expression(tokens):
    if not tokens:
        raise SyntaxError("Expression vide.")

    output = []
    while tokens:
        token = tokens.pop(0)
        kind, value = token

        if kind == 'DISTANCE':
            if not tokens or tokens.pop(0)[0] != 'LBRACE':
                raise SyntaxError("Accolade ouvrante '{' attendue après DISTANCE")
            arg1 = tokens.pop(0)
            if arg1[0] != 'SPATIAL_IDENT':
                raise SyntaxError("Premier argument de DISTANCE invalide.")
            if not tokens or tokens.pop(0)[0] != 'COMMA':
                raise SyntaxError("Virgule attendue entre les deux points")
            arg2 = tokens.pop(0)
            if arg2[0] != 'SPATIAL_IDENT':
                raise SyntaxError("Deuxième argument de DISTANCE invalide.")
            if not tokens or tokens.pop(0)[0] != 'RBRACE':
                raise SyntaxError("Accolade fermante '}' attendue après arguments DISTANCE")
            output.append(('DISTANCE_CALL', arg1[1], arg2[1]))
        elif kind == 'LIST_NAME' and tokens and tokens[0][0] == 'LBRACE':
            tokens.pop(0)  # consume LBRACE
            index_token = tokens.pop(0)
            if index_token[0] != 'NUMBER':
                raise SyntaxError("Index de liste invalide.")
            if not tokens or tokens.pop(0)[0] != 'RBRACE':
                raise SyntaxError("'}' attendu après l'index.")
            output.append(('LIST_ACCESS', value, int(index_token[1])))
        elif kind == 'LIST_NAME':
            output.append(token)
        elif kind in ('NUMBER', 'REAL', 'IDENT', 'CHAR', 'SPATIAL_IDENT'):
            output.append(token)
        elif kind in ('PLUS', 'MINUS', 'MULT', 'DIV'):
            output.append(token)
        elif kind == 'SEMICOLON':
            break
        else:
            raise SyntaxError(f"Token inattendu dans l'expression : {token}")

    return ('expression', output)

def eval_expression(expr_tokens):
    stack = []

    def resolve(token):
        kind = token[0]
        value = token[1]
        if kind == 'NUMBER':
            return int(value)
        elif kind == 'REAL':
            return float(value)
        elif kind == 'CHAR':
            return str(value.strip('"'))
        elif kind == 'BOOL_TRUE':
            return True
        elif kind == 'BOOL_FALSE':
            return False
        elif token[0] == 'LIST_ACCESS':
            varname, index = token[1], token[2]
            if varname not in variables:
                raise NameError(f"Variable {varname} non définie.")
            try:
                return variables[varname][index]
            except IndexError:
                raise IndexError(f"Index {index} hors limites pour la liste {varname}.")
        elif kind == 'LIST_NAME':
            varname = token[1]
            if varname not in variables:
                raise NameError(f"Liste {varname} non définie.")
            return variables[varname]
        elif kind == 'IDENT':
            """if value not in variables:
                raise NameError(f"Variable {value} non définie.")"""
            return variables[value]
        elif kind == 'SPATIAL_IDENT':
            if value not in variables:
                raise NameError(f"Variable spatiale {value} non définie.")
            return variables[value]
        elif token[0] == 'DISTANCE_CALL':
            name1, name2 = token[1], token[2]
            if name1 not in variables or name2 not in variables:
                raise NameError("Un des points n'est pas défini.")
            pt1 = variables[name1]
            pt2 = variables[name2]
            if not isinstance(pt1, Point) or not isinstance(pt2, Point):
                raise TypeError("DISTANCE ne peut être utilisé qu'entre deux POINTS.")
            return pt1.distance_to(pt2)
        else:
            raise ValueError(f"Token invalide dans resolve : {token}")

    i = 0
    while i < len(expr_tokens):
        token = expr_tokens[i]
        if token[0] in ('NUMBER', 'REAL', 'CHAR', 'IDENT', 'LIST_ACCESS', 'LIST_NAME', 'SPATIAL_IDENT', 'DISTANCE_CALL'):
            stack.append(resolve(token))
        elif token[0] in ('PLUS', 'MINUS', 'MULT', 'DIV'):
            op = token[0]
            a = stack.pop()
            b = resolve(expr_tokens[i + 1])
            if op == 'PLUS':
                if isinstance(a, str) or isinstance(b, str):
                    stack.append(str(a) + str(b))
                else:
                    stack.append(a + b)
            elif op == 'MINUS':
                stack.append(a - b)
            elif op == 'MULT':
                stack.append(a * b)
            elif op == 'DIV':
                stack.append(a / b)  # division entière
            i += 1  # saute l'opérande suivant qu'on vient de consommer
        i += 1

    """if len(stack) != 1:
        raise ValueError("Expression invalide.")"""
    return stack[0]

# Parser
def parse(tokens):
    tokens = list(tokens)
    if not tokens:
        return None

    if tokens[0][0] == 'INT':
        # Déclaration de variable
        _, _ = tokens.pop(0)
        ident = tokens.pop(0)
        assign = tokens.pop(0)
        value = tokens.pop(0)
        semicolon = tokens.pop(0)
        if assign[0] != 'ASSIGN' or value[0] != 'NUMBER' or semicolon[0] != 'SEMICOLON':
            raise SyntaxError("Syntaxe invalide dans la déclaration de variable.")
        return ('declare_var', ident[1], int(value[1]))
    
    elif tokens[0][0] == 'FLOAT':
        # Déclaration de variable
        _, _ = tokens.pop(0)
        ident = tokens.pop(0)
        assign = tokens.pop(0)
        value = tokens.pop(0)
        semicolon = tokens.pop(0)
        if assign[0] != 'ASSIGN' or value[0] != 'REAL' or semicolon[0] != 'SEMICOLON':
            raise SyntaxError("Syntaxe invalide dans la déclaration de variable.")
        return ('declare_var', ident[1], float(value[1]))
    
    elif tokens[0][0] == 'STR':
        # Déclaration de chaîne
        tokens.pop(0)
        ident = tokens.pop(0)
        assign = tokens.pop(0)
        value = tokens.pop(0)
        semicolon = tokens.pop(0)
        if assign[0] != 'ASSIGN' or value[0] != 'CHAR' or semicolon[0] != 'SEMICOLON':
            raise SyntaxError("Syntaxe invalide dans la déclaration de chaîne.")
        return ('declare_var', ident[1], value[1].strip('"'))
    
    elif tokens[0][0] == 'BOOL':
        # Déclaration de booléen
        tokens.pop(0)
        ident = tokens.pop(0)
        assign = tokens.pop(0)
        value = tokens.pop(0)
        semicolon = tokens.pop(0)
        if assign[0] != 'ASSIGN' or value[0] not in ('BOOL_TRUE', 'BOOL_FALSE') or semicolon[0] != 'SEMICOLON':
            raise SyntaxError("Syntaxe invalide dans la déclaration booléenne.")
        return ('declare_var', ident[1], value[0] == 'BOOL_TRUE')
    
    elif tokens[0][0] == 'LIST':
        # Déclaration de liste
        tokens.pop(0)  # LIST
        list_name = tokens.pop(0)
        assign = tokens.pop(0)
        list_open = tokens.pop(0)
        if list_name[0] != 'LIST_NAME' or assign[0] != 'ASSIGN' or list_open[0] != 'LBRACK':
            raise SyntaxError("Syntaxe invalide dans la déclaration de liste.")
        # Parse les éléments de la liste
        elements = []
        while tokens and tokens[0][0] != 'RBRACK':
            token = tokens.pop(0)
            if token[0] == 'CHAR':
                elements.append(token[1].strip('"'))
            elif token[0] == 'NUMBER':
                elements.append(int(token[1]))
            elif token[0] == 'REAL':
                elements.append(float(token[1]))
            elif token[0] == 'BOOL_TRUE':
                elements.append(True)
            elif token[0] == 'BOOL_FALSE':
                elements.append(False)
            elif token[0] == 'COMMA':
                continue
            else:
                raise SyntaxError(f"Type d'élément non pris en charge dans la liste : {token}")
        if not tokens or tokens.pop(0)[0] != 'RBRACK':
            raise SyntaxError("Liste non fermée.")
        if not tokens or tokens.pop(0)[0] != 'SEMICOLON':
            raise SyntaxError("Point-virgule attendu après la liste.")
        return ('declare_var', list_name[1], elements)
    
    elif tokens[0][0] == 'POINT':
        tokens.pop(0)  # POINT
        ident = tokens.pop(0)
        assign = tokens.pop(0)
        lparen = tokens.pop(0)
        lat = tokens.pop(0)
        comma = tokens.pop(0)
        lon = tokens.pop(0)
        rparen = tokens.pop(0)
        epsg_token = tokens.pop(0)
        semicolon = tokens.pop(0)
        if (
            ident[0] != 'SPATIAL_IDENT'
            or assign[0] != 'ASSIGN'
            or lparen[0] != 'LPAREN'
            or comma[0] != 'COMMA'
            or rparen[0] != 'RPAREN'
            or lat[0] not in ('REAL', 'NUMBER')
            or lon[0] not in ('REAL', 'NUMBER')
            or epsg_token[0] != 'EPSG'
            or semicolon[0] != 'SEMICOLON'
        ):
            raise SyntaxError("Syntaxe invalide dans la déclaration de point.")
        lat_val = float(lat[1]) if lat[0] == 'REAL' else int(lat[1])
        lon_val = float(lon[1]) if lon[0] == 'REAL' else int(lon[1])
        epsg_code = int(epsg_token[1][1:-1])
        return ('declare_var', ident[1], Point(lat_val, lon_val, epsg_code))
    
    elif tokens[0][0] == 'LINE':
        tokens.pop(0)
        ident = tokens.pop(0)
        assign = tokens.pop(0)
        if ident[0] != 'SPATIAL_IDENT' or assign[0] != 'ASSIGN':
            raise SyntaxError("Syntaxe invalide dans la déclaration de LINE.")
        if not tokens or tokens.pop(0)[0] != 'LPAREN':
            raise SyntaxError("'(' attendu après <- dans LINE.")
        points = []
        while tokens and tokens[0][0] != 'RPAREN':
            if tokens[0][0] == 'LPAREN':
                tokens.pop(0)
                lat = tokens.pop(0)
                if tokens.pop(0)[0] != 'COMMA':
                    raise SyntaxError("Virgule attendue dans la coordonnée.")
                lon = tokens.pop(0)
                if tokens.pop(0)[0] != 'RPAREN':
                    raise SyntaxError("')' attendu après coordonnée.")
                if lat[0] not in ('NUMBER', 'REAL') or lon[0] not in ('NUMBER', 'REAL'):
                    raise SyntaxError("Coordonnées invalides dans LINE.")
                lat_val = float(lat[1]) if lat[0] == 'REAL' else int(lat[1])
                lon_val = float(lon[1]) if lon[0] == 'REAL' else int(lon[1])
                epsg = int(tokens[-2][-1][1:-1])
                points.append(Point(lat_val, lon_val, epsg))
            elif tokens[0][0] == 'SPATIAL_IDENT':
                ref = tokens.pop(0)[1]
                """if ref not in variables or not isinstance(variables[ref], Point):
                    raise SyntaxError(f"Variable {ref} non définie ou non de type POINT.")"""
                points.append(variables[ref])
            if tokens and tokens[0][0] == 'COMMA':
                tokens.pop(0)
        if not tokens or tokens.pop(0)[0] != 'RPAREN':
            raise SyntaxError("')' attendu pour fermer la LINE.")
        epsg_token = tokens.pop(0)
        if epsg_token[0] != 'EPSG':
            raise SyntaxError("Code EPSG attendu après les coordonnées de la LINE.")
        if len(points) < 2:
            raise SyntaxError("Une LINE doit contenir au moins deux points.")
        if not tokens or tokens.pop(0)[0] != 'SEMICOLON':
            raise SyntaxError("Point-virgule attendu après la LINE.")
        epsg_code = int(epsg_token[1][1:-1])
        return ('declare_var', ident[1], Line(points, epsg_code))

    elif tokens[0][0] == 'POLYGON':
        tokens.pop(0)
        ident = tokens.pop(0)
        assign = tokens.pop(0)
        if ident[0] != 'SPATIAL_IDENT' or assign[0] != 'ASSIGN':
            raise SyntaxError("Syntaxe invalide dans la déclaration de POLYGON.")
        if not tokens or tokens.pop(0)[0] != 'LPAREN':
            raise SyntaxError("'(' attendu après <- dans POLYGON.")
        points = []
        while tokens and tokens[0][0] != 'RPAREN':
            if tokens[0][0] == 'LPAREN':
                tokens.pop(0)
                x = tokens.pop(0)
                if tokens.pop(0)[0] != 'COMMA':
                    raise SyntaxError("Virgule attendue dans la coordonnée.")
                y = tokens.pop(0)
                if tokens.pop(0)[0] != 'RPAREN':
                    raise SyntaxError("')' attendu après coordonnée.")
                if x[0] not in ('NUMBER', 'REAL') or y[0] not in ('NUMBER', 'REAL'):
                    raise SyntaxError("Coordonnées invalides dans POLYGON.")
                x_val = float(x[1]) if x[0] == 'REAL' else int(x[1])
                y_val = float(y[1]) if y[0] == 'REAL' else int(y[1])
                epsg = int(tokens[-2][-1][1:-1])
                points.append(Point(x_val, y_val, epsg))
            elif tokens[0][0] == 'SPATIAL_IDENT':
                ref = tokens.pop(0)[1]
                """if ref not in variables or not isinstance(variables[ref], Point):
                    raise SyntaxError(f"Variable {ref} non définie ou non de type POINT.")"""
                points.append(variables[ref])
            if tokens and tokens[0][0] == 'COMMA':
                tokens.pop(0)
        if not tokens or tokens.pop(0)[0] != 'RPAREN':
            raise SyntaxError("')' attendu pour fermer le POLYGON.")
        epsg_token = tokens.pop(0)
        if epsg_token[0] != 'EPSG':
            raise SyntaxError("Code EPSG attendu après les coordonnées de la LINE.")
        if len(points) < 3:
            raise SyntaxError("Un POLYGON doit contenir au moins trois sommets.")
        if not tokens or tokens.pop(0)[0] != 'SEMICOLON':
            raise SyntaxError("Point-virgule attendu après le POLYGON.")
        epsg_code = int(epsg_token[1][1:-1])
        return ('declare_var', ident[1], Polygon(points, epsg_code))
    
    elif tokens[0][0] == 'MAP':
        tokens.pop(0)
        target = tokens.pop(0)
        if target[0] not in ('SPATIAL_IDENT',):
            raise SyntaxError("MAP attend un identifiant spatial.")
        if tokens.pop(0)[0] != 'SEMICOLON':
            raise SyntaxError("Point-virgule attendu après MAP.")
        return ('map_draw', target[1])
    
    elif tokens[0][0] == 'PRINT':
        tokens.pop(0)  # Supprime le PRINT
        expr = parse_expression(tokens)
        return ('print_expr', expr)
    
    """else:
        raise SyntaxError("Instruction inconnue.")"""

# Interpréteur
def execute(ast):
    if ast is None:
        return
    action = ast[0]
    if action == 'declare_var':
        _, var_name, value = ast
        variables[var_name] = value
    elif action == 'print_value':
        _, value_type, value = ast
        if value_type == 'IDENT':
            if value in variables:
                print(variables[value])
            else:
                print(f"Erreur : Variable {value} non définie.")
        elif value_type == 'NUMBER':
            print(value)
        elif value_type == 'REAL':
            print(value)
        elif value_type == 'CHAR':
            print(value)
    elif action == 'print_expr':
        _, expr = ast
        val = eval_expression(expr[1])
        if isinstance(val, tuple):
            # Affiche un POINT (x, y)
            print(f"({val[0]}, {val[1]})")
        elif isinstance(val, list) and all(isinstance(p, tuple) and len(p) == 2 for p in val):
            # Affiche une LINE ((x1, y1), (x2, y2), ...)
            coords = ', '.join(f"({p[0]}, {p[1]})" for p in val)
            print(f"({coords})")
        else:
            print(val)
    elif action == 'map_draw':
        varname = ast[1]
        if varname not in variables:
            raise NameError(f"Variable {varname} non définie.")

        data = variables[varname]

        import folium
        import webbrowser
        from pyproj import Transformer

        if isinstance(data, Point):
            transformer = Transformer.from_crs("EPSG:"+str(data.epsg), "EPSG:4326", always_xy=True)
            coords = transformer.transform(data.x, data.y)
            m = folium.Map(location=coords, zoom_start=12)
            folium.Marker(location=coords, popup=varname).add_to(m)
            m.fit_bounds(coords)

        elif isinstance(data, Line):
            transformer = Transformer.from_crs("EPSG:"+str(data.epsg), "EPSG:4326", always_xy=True)
            coords = [(p.x, p.y) for p in data.points]
            coords = [transformer.transform(x, y) for x, y in coords]
            #print("--------")
            #print(coords)
            #print("--------")
            m = folium.Map(location=coords[0], zoom_start=12)
            folium.PolyLine(locations=coords, color="blue", popup=varname).add_to(m)
            m.fit_bounds(coords)

        elif isinstance(data, Polygon):
            transformer = Transformer.from_crs("EPSG:"+str(data.epsg), "EPSG:4326", always_xy=True)
            coords = [(p.x, p.y) for p in data.points]
            if coords[0] != coords[-1]:
                coords.append(coords[0])
            m = folium.Map(location=coords[0], zoom_start=12)
            folium.Polygon(locations=coords, color="green", popup=varname, fill=True, fill_opacity=0.3).add_to(m)
            m.fit_bounds(coords)
        else:
            raise ValueError("Donnée non géographique.")
        m.save("map.html")
        webbrowser.open("map.html")
    
def run_file(filename):
    with open(filename, 'r') as file:
        code = file.read()
        run(code)

# Programme principal
def run(code):
    # Sépare les instructions par `;`, en conservant le `;` pour la syntaxe
    instructions = [instr.strip() + ";" for instr in code.split(";") if instr.strip()]
    
    for instr in instructions:
        tokens = tokenize(instr)
        ast = parse(tokens)
        execute(ast)

# Exemple d'utilisation
#run("INT a# <- 42;")
#run("PRINT a#;")

if __name__ == '__main__':
    import sys
    if len(sys.argv) != 2:
        print("Usage : python monlangage.py test.obs")
    else:
        run_file(sys.argv[1])