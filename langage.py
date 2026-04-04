import re, os, sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from objets import Point, Line, Polygon
from data import token_specification, args

# Table des variables
variables = {}

tok_regex = '|'.join('(?P<%s>%s)' % pair for pair in token_specification)
get_token = re.compile(tok_regex).match

def tokenize(code):
    pos = 0
    mo = get_token(code, pos)
    while mo:
        kind = mo.lastgroup
        value = mo.group()
        if kind not in ('SKIP', 'NEWLINE', 'COMMENT'):
            #print(f"[DEBUG TOKEN] {kind}: {value}")  
            yield (kind, value)
        pos = mo.end()
        mo = get_token(code, pos)
    if pos != len(code):
        raise RuntimeError(f'Unexpected character {code[pos]}')

def parse_expression(tokens):
    """Parse une expression à partir de la liste de tokens. Retourne une structure représentant l'expression.
    Les expressions peuvent être des appels de fonctions (DISTANCE, LENGTH, PERIMETER, AREA) ou des opérations arithmétiques/logiques.
    La fonction gère également l'accès aux éléments de liste (LIST_NAME{index}) et les identifiants spatiaux (SPATIAL_IDENT)."""
    if not tokens:
        raise SyntaxError("Expression vide.")

    output = []
    while tokens:
        kind, value = tokens[0]
        if kind == 'SEMICOLON':
            break
        token = tokens.pop(0)

        if token[0] in args: # args -> data.py
            expected_args = args[token[0]]

            # Vérifie si accolade ouvrante présente
            if not tokens or tokens.pop(0)[0] != 'LBRACE':
                raise SyntaxError("Accolade ouvrante '{' attendue après " + kind)
            
            arg_tokens = []

            # Boucle sur les arguments attendus
            for i, expected in enumerate(expected_args):
                # Récupère l'argument
                if not tokens:
                    raise SyntaxError("Argument manquant pour " + kind)
                arg = tokens.pop(0)
                # Vérifie le type de l'argument
                if arg[0] != expected:
                    raise SyntaxError(f"Argument de type {expected} attendu pour {kind}, mais {arg[0]} trouvé.")
                arg_tokens.append(arg)
                # Si ce n'est pas le dernier argument, attente d'une virgule
                if i < len(expected_args) - 1:
                    if not tokens or tokens[0][0] != 'COMMA':
                        raise SyntaxError("Virgule attendue entre les arguments de " + kind)
                    tokens.pop(0)  # Consommer la virgule
            # Vérifie si accolade fermante présente
            if not tokens or tokens.pop(0)[0] != 'RBRACE':
                raise SyntaxError("Accolade fermante '}' attendue après les arguments de " + kind)
            
            # Ajoute l'appel de fonction à la sortie
            call_name = f"{kind}_CALL"
            call_args = [arg[1] for arg in arg_tokens]
            output.append((call_name, *call_args))

        # if kind == 'DISTANCE':
        #     if not tokens or tokens.pop(0)[0] != 'LBRACE':
        #         raise SyntaxError("Accolade ouvrante '{' attendue après DISTANCE")
        #     arg1 = tokens.pop(0)
        #     if arg1[0] != 'SPATIAL_IDENT':
        #         raise SyntaxError("Premier argument de DISTANCE invalide.")
        #     if not tokens or tokens.pop(0)[0] != 'COMMA':
        #         raise SyntaxError("Virgule attendue entre les deux points")
        #     arg2 = tokens.pop(0)
        #     if arg2[0] != 'SPATIAL_IDENT':
        #         raise SyntaxError("Deuxième argument de DISTANCE invalide.")
        #     if not tokens or tokens.pop(0)[0] != 'RBRACE':
        #         raise SyntaxError("Accolade fermante '}' attendue après arguments DISTANCE")
        #     output.append(('DISTANCE_CALL', arg1[1], arg2[1]))
        # elif kind == 'LENGTH':
        #     if not tokens or tokens.pop(0)[0] != 'LBRACE':
        #         raise SyntaxError("Accolade ouvrante '{' attendue après LENGTH")
        #     arg1 = tokens.pop(0)
        #     if arg1[0] != 'SPATIAL_IDENT':
        #         raise SyntaxError("Premier argument de LENGTH invalide.")
        #     if not tokens or tokens.pop(0)[0] != 'RBRACE':
        #         raise SyntaxError("Accolade fermante '}' attendue après arguments LENGTH")
        #     output.append(('LENGTH_CALL', arg1[1]))
        # elif kind == 'PERIMETER':
        #     if not tokens or tokens.pop(0)[0] != 'LBRACE':
        #         raise SyntaxError("Accolade ouvrante '{' attendue après PERIMETER")
        #     arg1 = tokens.pop(0)
        #     if arg1[0] != 'SPATIAL_IDENT':
        #         raise SyntaxError("Premier argument de PERIMETER invalide.")
        #     if not tokens or tokens.pop(0)[0] != 'RBRACE':
        #         raise SyntaxError("Accolade fermante '}' attendue après arguments PERIMETER")
        #     output.append(('PERIMETER_CALL', arg1[1]))
        # elif kind == 'AREA':
        #     if not tokens or tokens.pop(0)[0] != 'LBRACE':
        #         raise SyntaxError("Accolade ouvrante '{' attendue après AREA")
        #     arg1 = tokens.pop(0)
        #     if arg1[0] != 'SPATIAL_IDENT':
        #         raise SyntaxError("Premier argument de AREA invalide.")
        #     if not tokens or tokens.pop(0)[0] != 'RBRACE':
        #         raise SyntaxError("Accolade fermante '}' attendue après arguments AREA")
        #     output.append(('AREA_CALL', arg1[1]))
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
        elif kind in ('NUMBER', 'REAL', 'IDENT', 'CHAR', 'SPATIAL_IDENT', 'BOOL_TRUE', 'BOOL_FALSE'):
            output.append(token)
        elif kind in ('PLUS', 'MINUS', 'MULT', 'DIV', 'EQ', 'GT', 'LT', 'GE', 'LE', 'NE', 'LPAREN', 'RPAREN'):
            output.append(token)
        elif kind == 'SEMICOLON':
            break
        else:
            raise SyntaxError(f"Token inattendu dans l'expression : {token}")

    return ('expression', output)

def eval_expression(expr_tokens):
    """Évalue une expression avec gestion des priorités et des parenthèses"""
    
    def resolve_value(token):
        """Résout une valeur primitive (nombre, variable, appel de fonction)"""
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
            if value not in variables:
                raise NameError(f"Variable {value} non définie.")
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
        elif token[0] == 'LENGTH_CALL':
            name = token[1]
            if name not in variables:
                raise NameError("La ligne n'est pas définie.")
            line = variables[name]
            return line.length()
        elif token[0] == 'PERIMETER_CALL':
            name = token[1]
            if name not in variables:
                raise NameError("Le polygone n'est pas défini.")
            polygon = variables[name]
            return polygon.perimeter()
        elif token[0] == 'AREA_CALL':
            name = token[1]
            if name not in variables:
                raise NameError("Le polygone n'est pas défini.")
            polygon = variables[name]
            return polygon.area()
        else:
            raise ValueError(f"Token invalide dans resolve_value : {token}")

    pos = [0]  # Position actuelle dans les tokens (par référence pour pouvoir la modifier)

    def current():
        """Retourne le token actuel"""
        if pos[0] < len(expr_tokens):
            return expr_tokens[pos[0]]
        return None

    def consume():
        """Consomme le token actuel et passe au suivant"""
        token = current()
        pos[0] += 1
        return token

    def eval_primary():
        """Évalue un terme primaire (nombre, variable, parenthèses)"""
        token = current()
        if not token:
            raise SyntaxError("Expression vide ou incomplète.")
        
        if token[0] == 'LPAREN':
            consume()  # consume '('
            result = eval_comparison()  # parse l'expression à l'intérieur des parenthèses
            if not current() or current()[0] != 'RPAREN':
                raise SyntaxError("')' attendu.")
            consume()  # consume ')'
            return result
        elif token[0] in ('NUMBER', 'REAL', 'CHAR', 'BOOL_TRUE', 'BOOL_FALSE', 
                          'IDENT', 'SPATIAL_IDENT', 'LIST_NAME', 'LIST_ACCESS',
                          'DISTANCE_CALL', 'LENGTH_CALL', 'PERIMETER_CALL', 'AREA_CALL'):
            consume()
            return resolve_value(token)
        elif token[0] == 'MINUS':
            # Gère l'unaire moins
            consume()
            value = eval_primary()
            return -value
        elif token[0] == 'PLUS':
            # Gère l'unaire plus
            consume()
            return eval_primary()
        else:
            raise SyntaxError(f"Token inattendu {token[0]}")

    def eval_multiplicative():
        """Évalue les opérations * et /"""
        result = eval_primary()
        while current() and current()[0] in ('MULT', 'DIV'):
            op = consume()
            right = eval_primary()
            if op[0] == 'MULT':
                result = result * right
            elif op[0] == 'DIV':
                result = result / right
        return result

    def eval_additive():
        """Évalue les opérations + et -"""
        result = eval_multiplicative()
        while current() and current()[0] in ('PLUS', 'MINUS'):
            op = consume()
            right = eval_multiplicative()
            if op[0] == 'PLUS':
                if isinstance(result, str) or isinstance(right, str):
                    result = str(result) + str(right)
                else:
                    result = result + right
            elif op[0] == 'MINUS':
                result = result - right
        return result

    def eval_comparison():
        """Évalue les opérations de comparaison"""
        result = eval_additive()
        while current() and current()[0] in ('EQ', 'GT', 'LT', 'GE', 'LE', 'NE'):
            op = consume()
            right = eval_additive()
            if op[0] == 'EQ':
                result = result == right
            elif op[0] == 'GT':
                result = result > right
            elif op[0] == 'LT':
                result = result < right
            elif op[0] == 'GE':
                result = result >= right
            elif op[0] == 'LE':
                result = result <= right
            elif op[0] == 'NE':
                result = result != right
        return result

    result = eval_comparison()
    if pos[0] < len(expr_tokens):
        raise SyntaxError(f"Tokens non consommés après l'expression : {expr_tokens[pos[0]:]}")
    return result


# Parser
def parse(tokens):
    if not tokens:
        return None

    types = {
        "INT": ("variable_unique", ["ASSIGN", "expression", "SEMICOLON"]),
        "FLOAT": ("variable_unique", ["ASSIGN", "expression", "SEMICOLON"]),
        "STR": ("variable_unique", ["ASSIGN", "expression", "SEMICOLON"]),
        "BOOL": ("variable_unique", ["ASSIGN", "expression", "SEMICOLON"])
    }

    def token_matches(expected, actual):
        return actual == expected or (isinstance(expected, tuple) and actual in expected)

    if tokens[0][0] in types:
        stmt_type = tokens[0][0]
        type_name, expected_tokens = types[stmt_type]
        if type_name == "variable_unique":
            tokens.pop(0)  # consume type
            ident = tokens.pop(0)
            assign = tokens.pop(0)
            if assign[0] != 'ASSIGN':
                raise SyntaxError("Syntaxe invalide dans la déclaration de variable.")

            expr = parse_expression(tokens)
            if not tokens or tokens.pop(0)[0] != 'SEMICOLON':
                raise SyntaxError("Syntaxe invalide dans la déclaration de variable.")

            value = eval_expression(expr[1])
            if stmt_type == 'INT':
                if not isinstance(value, (int, float)):
                    raise TypeError("Un INT doit être une expression numérique.")
                return ('declare_var', ident[1], int(float(value)))
            elif stmt_type == 'FLOAT':
                if not isinstance(value, (int, float)):
                    raise TypeError("Un FLOAT doit être une expression numérique.")
                return ('declare_var', ident[1], float(value))
            elif stmt_type == 'STR':
                return ('declare_var', ident[1], str(value))
            elif stmt_type == 'BOOL':
                if not isinstance(value, bool):
                    raise TypeError("Un BOOL doit être une expression booléenne.")
                return ('declare_var', ident[1], value)

    # if tokens[0][0] == 'INT':
    #     # Déclaration de variable
    #     _, _ = tokens.pop(0)
    #     ident = tokens.pop(0)
    #     assign = tokens.pop(0)
    #     value = tokens.pop(0)
    #     semicolon = tokens.pop(0)
    #     if assign[0] != 'ASSIGN' or value[0] != 'NUMBER' or semicolon[0] != 'SEMICOLON':
    #         raise SyntaxError("Syntaxe invalide dans la déclaration de variable.")
    #     return ('declare_var', ident[1], int(value[1]))

    
    # elif tokens[0][0] == 'FLOAT':
    #     # Déclaration de variable
    #     _, _ = tokens.pop(0)
    #     ident = tokens.pop(0)
    #     assign = tokens.pop(0)
    #     value = tokens.pop(0)
    #     semicolon = tokens.pop(0)
    #     if assign[0] != 'ASSIGN' or value[0] != 'REAL' or semicolon[0] != 'SEMICOLON':
    #         raise SyntaxError("Syntaxe invalide dans la déclaration de variable.")
    #     return ('declare_var', ident[1], float(value[1]))
    
    # elif tokens[0][0] == 'STR':
    #     # Déclaration de chaîne
    #     tokens.pop(0)
    #     ident = tokens.pop(0)
    #     assign = tokens.pop(0)
    #     value = tokens.pop(0)
    #     semicolon = tokens.pop(0)
    #     if assign[0] != 'ASSIGN' or value[0] != 'CHAR' or semicolon[0] != 'SEMICOLON':
    #         raise SyntaxError("Syntaxe invalide dans la déclaration de chaîne.")
    #     return ('declare_var', ident[1], value[1].strip('"'))
    
    # elif tokens[0][0] == 'BOOL':
    #     # Déclaration de booléen
    #     tokens.pop(0)
    #     ident = tokens.pop(0)
    #     assign = tokens.pop(0)
    #     value = tokens.pop(0)
    #     semicolon = tokens.pop(0)
    #     if assign[0] != 'ASSIGN' or value[0] not in ('BOOL_TRUE', 'BOOL_FALSE') or semicolon[0] != 'SEMICOLON':
    #         raise SyntaxError("Syntaxe invalide dans la déclaration booléenne.")
    #     return ('declare_var', ident[1], value[0] == 'BOOL_TRUE')
    
    if tokens[0][0] == 'LIST':
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
                points.append((lat_val, lon_val))
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
        resolved_points = []
        for p in points:
            if isinstance(p, tuple):
                resolved_points.append(Point(p[0], p[1], epsg_code))
            else:
                resolved_points.append(p)
        return ('declare_var', ident[1], Line(resolved_points, epsg_code))

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
                points.append((x_val, y_val))
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
        resolved_points = []
        for p in points:
            if isinstance(p, tuple):
                resolved_points.append(Point(p[0], p[1], epsg_code))
            else:
                resolved_points.append(p)
        return ('declare_var', ident[1], Polygon(resolved_points, epsg_code))
    
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
        if not tokens or tokens.pop(0)[0] != 'SEMICOLON':
            raise SyntaxError("Point-virgule attendu après PRINT.")
        return ('print_expr', expr)
    
    elif tokens[0][0] == 'IF':
        return parse_if(tokens)
    
    """else:
        raise SyntaxError("Instruction inconnue.")"""

def parse_if(tokens):
    tokens.pop(0)  # consume IF
    if tokens.pop(0)[0] != 'LPAREN':
        raise SyntaxError("'(' attendu après IF.")
    # Collect tokens for condition until RPAREN
    cond_tokens = []
    paren_count = 1
    while tokens and paren_count > 0:
        token = tokens.pop(0)
        if token[0] == 'LPAREN':
            paren_count += 1
        elif token[0] == 'RPAREN':
            paren_count -= 1
        if paren_count > 0:
            cond_tokens.append(token)
    condition = ('expression', cond_tokens)
    if tokens.pop(0)[0] != 'THEN':
        raise SyntaxError("THEN attendu après la condition.")
    
    # Parse le bloc THEN
    then_block = []
    while tokens and tokens[0][0] not in ('ELSEIF', 'ELSE', 'END'):
        stmt = parse(tokens)
        if stmt:
            then_block.append(stmt)
    
    elif_branches = []
    else_block = None
    
    while tokens and tokens[0][0] != 'END':
        if tokens[0][0] == 'ELSEIF':
            tokens.pop(0)  # consume ELSEIF
            if tokens.pop(0)[0] != 'LPAREN':
                raise SyntaxError("'(' attendu après ELSEIF.")
            elif_cond_tokens = []
            paren_count = 1
            while tokens and paren_count > 0:
                token = tokens.pop(0)
                if token[0] == 'LPAREN':
                    paren_count += 1
                elif token[0] == 'RPAREN':
                    paren_count -= 1
                if paren_count > 0:
                    elif_cond_tokens.append(token)
            elif_condition = ('expression', elif_cond_tokens)
            if tokens.pop(0)[0] != 'THEN':
                raise SyntaxError("THEN attendu après la condition ELSEIF.")
            elif_block = []
            while tokens and tokens[0][0] not in ('ELSEIF', 'ELSE', 'END'):
                stmt = parse(tokens)
                if stmt:
                    elif_block.append(stmt)
            elif_branches.append((elif_condition, elif_block))
        elif tokens[0][0] == 'ELSE':
            tokens.pop(0)  # consume ELSE
            else_block = []
            while tokens and tokens[0][0] != 'END':
                stmt = parse(tokens)
                if stmt:
                    else_block.append(stmt)
            break
        else:
            raise SyntaxError("ELSEIF, ELSE ou END attendu.")
    
    if not tokens or tokens.pop(0)[0] != 'END':
        raise SyntaxError("END attendu pour fermer le IF.")
    
    return ('if_stmt', condition, then_block, elif_branches, else_block)

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
    elif action == 'if_stmt':
        _, condition, then_block, elif_branches, else_block = ast
        cond_val = eval_expression(condition[1])
        if cond_val:
            for stmt in then_block:
                execute(stmt)
        else:
            executed = False
            for elif_cond, elif_block in elif_branches:
                elif_val = eval_expression(elif_cond[1])
                if elif_val:
                    for stmt in elif_block:
                        execute(stmt)
                    executed = True
                    break
            if not executed and else_block:
                for stmt in else_block:
                    execute(stmt)
    
def run_file(filename):
    from pathlib import Path
    filename = Path(filename)
    if not filename.exists():
        raise FileNotFoundError(f"Fichier non trouvé : {filename}")
    with filename.open('r', encoding='utf-8') as file:
        code = file.read()
        run(code)

# Programme principal
def run(code):
    tokens = [t for t in tokenize(code) if t[0] not in ('SKIP', 'NEWLINE', 'MISMATCH')]
    while tokens:
        ast = parse(tokens)
        execute(ast)

# Exemple d'utilisation
#run("INT a# <- 42;")
#run("PRINT a#;")

def main(argv=None):
    import sys
    from pathlib import Path
    if argv is None:
        argv = sys.argv

    if len(argv) != 2:
        print("Usage : obsidian <fichier.obs>")
        sys.exit(1)

    source_path = Path(argv[1]).expanduser()
    if not source_path.exists():
        print(f"Fichier non trouvé : {source_path}")
        sys.exit(1)

    try:
        run_file(source_path)
    except Exception as e:
        print(f"Erreur d'exécution : {e}")
        sys.exit(1)

    sys.exit(0)

if __name__ == '__main__':
    main()