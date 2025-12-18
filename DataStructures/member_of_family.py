# Related Members in a Family Tree

# Create a data structure to represent people and their family relationships,
# and then write an algorithm that would determine if two people are related
# (i.e. genetically, blood related, not by marriage).


class Person:
    def __init__(self, name: str):
        self.name = name
        self.parents = []
        self.children = []
        self.spouse = None

    def add_parent(self, parent: "Person"):
        self.parents.append(parent)
        parent.children.append(self)

    def add_spouse(self, spouse: "Person"):
        self.spouse = spouse
        spouse.spouse = self


Abuelita = Person("Abue")
Abuelito = Person("Abuelito")
Abuelita.add_spouse(Abuelito)

Lila = Person("Lila")
Oscar = Person("Oscar")
Lila.add_spouse(Oscar)
Lila.add_parent(Abuelita)
Lila.add_parent(Abuelito)

Oscarin = Person("Oscarin")
Oscarin.add_parent(Oscar)
Oscarin.add_parent(Lila)

Perlita = Person("Perlita")
Oscarin.add_spouse(Perlita)

Yo = Person("Yo")
Yo.add_parent(Oscar)
Yo.add_parent(Lila)

JoseLuis = Person("JL")
Lourdes = Person("Lourdes")
JoseLuis.add_spouse(Lourdes)

Amy = Person("Amy")
Yo.add_spouse(Amy)
Amy.add_parent(JoseLuis)
Amy.add_parent(Lourdes)

Olga = Person("Olga")
JoseLuis.add_parent(Olga)


# we want to know if parent is an ancestor of child
# at any point in the family tree
def is_ancestor(parent: Person, child: Person):
    if parent in child.parents:
        return True

    if child.parents:
        found = [is_ancestor(parent, p) for p in child.parents]
        return any(found)

    return False


print("Is Abuelita an ancestor of Yo?", is_ancestor(Abuelita, Yo))
print("Is Lila an ancestor of Oscarin?", is_ancestor(Lila, Oscarin))
print("Is Lila an ancestor of Perlita?", is_ancestor(Lila, Perlita))
print("Is Lila an ancestor of Amy?", is_ancestor(Lila, Amy))
print("Is Oscar an ancestor of Yo?", is_ancestor(Oscar, Yo))
print("Is Oscar an ancestor of Oscarin?", is_ancestor(Oscar, Oscarin))
print("Is Oscar an ancestor of Perlita?", is_ancestor(Oscar, Perlita))
print("Is Oscar an ancestor of Amy?", is_ancestor(Oscar, Amy))
print("Is Amy an ancestor of Yo?", is_ancestor(Amy, Yo))
print("Is JoseLuis an ancestor of Amy?", is_ancestor(JoseLuis, Amy))
print("Is Lourdes an ancestor of Amy?", is_ancestor(Lourdes, Amy))
print("Is Olga an ancestor of Amy?", is_ancestor(Olga, Amy))
print("Is Lila an ancestor of JoseLuis?", is_ancestor(Lila, JoseLuis))
