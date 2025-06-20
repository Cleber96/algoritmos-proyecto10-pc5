# src/m_tree/node.py
import numpy as np
from typing import List, Union, Optional, Tuple, Dict, Any
from src.common.models import Vector, SearchResult
from src.common.utils import euclidean_distance, get_distance_metric, logger

# Tipo de dato para representar un M-Entry (entrada en un nodo M-Tree)
# Contiene el punto de referencia, su radio de cobertura y el puntero al hijo.
MEntry = Tuple[Vector, float, Union['MTreeNode', Vector]] # (reference_point, radius, child_pointer)

class MTreeNode:
    """
    Clase base para los nodos del M-Tree.
    """
    def __init__(self, tree: 'MTree', is_leaf: bool = False):
        self.tree = tree
        self.is_leaf = is_leaf
        self.entries: List[MEntry] = []
        self.parent: Optional['MTreeNode'] = None

    def add_entry(self, entry: MEntry):
        """Añade una entrada al nodo."""
        self.entries.append(entry)

    def is_full(self) -> bool:
        """Verifica si el nodo ha alcanzado su capacidad máxima."""
        return len(self.entries) >= self.tree.max_children

    def is_underfull(self) -> bool:
        """Verifica si el nodo está por debajo de su capacidad mínima."""
        return len(self.entries) < self.tree.min_children

    def update_entry(self, old_entry: MEntry, new_entry: MEntry):
        """Actualiza una entrada existente en el nodo."""
        try:
            idx = self.entries.index(old_entry)
            self.entries[idx] = new_entry
        except ValueError:
            logger.warning(f"Intento de actualizar una entrada no existente en el nodo. Old: {old_entry[0].id}")

    def remove_entry(self, entry_to_remove: MEntry):
        """Elimina una entrada específica del nodo."""
        self.entries.remove(entry_to_remove)

    def get_reference_point(self, entry: MEntry) -> Vector:
        """Obtiene el punto de referencia de una entrada."""
        return entry[0]

    def get_radius(self, entry: MEntry) -> float:
        """Obtiene el radio de cobertura de una entrada."""
        return entry[1]

    def get_child_pointer(self, entry: MEntry) -> Union['MTreeNode', Vector]:
        """Obtiene el puntero al hijo (nodo o vector de datos) de una entrada."""
        return entry[2]

    def __len__(self):
        return len(self.entries)

    def __repr__(self):
        return f"MTreeNode(is_leaf={self.is_leaf}, num_entries={len(self.entries)})"


class MTreeInternalNode(MTreeNode):
    """
    Representa un nodo interno en el M-Tree.
    Contiene entradas que apuntan a otros nodos (hijos).
    """
    def __init__(self, tree: 'MTree'):
        super().__init__(tree, is_leaf=False)

    def get_child_node(self, entry: MEntry) -> 'MTreeNode':
        """Retorna el nodo hijo al que apunta la entrada."""
        child = self.get_child_pointer(entry)
        if not isinstance(child, MTreeNode):
            raise TypeError("Child pointer in internal node must be an MTreeNode.")
        return child

    def calculate_covering_radius(self, new_point: Vector) -> float:
        """
        Calcula el radio de cobertura mínimo que debe tener este nodo para incluir un nuevo punto.
        Esto es la distancia máxima desde cualquier punto en el nodo (o sus hijos) hasta el punto de referencia del nodo.
        En el contexto de la inserción, esto puede referirse a la expansión del radio de las entradas existentes
        para cubrir un nuevo punto que se va a insertar en uno de sus subárboles.
        """
        # Para un nodo interno, el radio de cobertura de una entrada MEntry(R, r, child) es r.
        # R es el punto de referencia del subárbol, r es el radio de cobertura de ese subárbol.
        # No es el radio que cubre todo el nodo en sí, sino el radio de la bola de cada entrada.
        # Esta función puede ser útil en la lógica de split, para recalcular radios.
        # Por ahora, es un placeholder.
        return 0.0 # Placeholder, la lógica de radio se maneja en las entradas individuales.

class MTreeLeafNode(MTreeNode):
    """
    Representa un nodo hoja en el M-Tree.
    Contiene entradas que apuntan directamente a los vectores de datos.
    """
    def __init__(self, tree: 'MTree'):
        super().__init__(tree, is_leaf=True)

    def get_data_vector(self, entry: MEntry) -> Vector:
        """Retorna el vector de datos al que apunta la entrada."""
        data_vector = self.get_child_pointer(entry) # En nodos hoja, child_pointer es el Vector
        if not isinstance(data_vector, Vector):
            raise TypeError("Child pointer in leaf node must be a Vector.")
        return data_vector