from typing import Collection, TypeVar

from src.pangu.language.domain_language import DomainLanguage, predicate

Nodes = TypeVar('Nodes', bound=Collection)
Property = TypeVar('Property')


class OpenKGLanguage(DomainLanguage):
    def __init__(self):
        super().__init__({}, start_types={Nodes})

    @predicate
    def AND(self, set1: Nodes, set2: Nodes) -> Nodes:
        pass

    @predicate
    def JOIN(self, p: Property, o: Nodes) -> Nodes:
        pass


if __name__ == '__main__':
    language = OpenKGLanguage()
    print(language.all_possible_productions())
