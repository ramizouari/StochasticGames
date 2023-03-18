//
// Created by ramizouari on 12/03/23.
//

#include "Variable.h"

std::strong_ordering Variable::operator<=>(const uint64_t &other) const {
    return id<=>other;
}

Variable *VariableFactory::create() {
    return variables.emplace_back(std::make_unique<Variable>(index++)).get();
}

VariableFactorySet::VariableFactorySet(std::set<Variable> predefined) : predefined(std::move(predefined)) {}

Variable *VariableFactorySet::create() {
    auto it=predefined.find(index);
    if(it!=predefined.end()) while(it!=predefined.end() && *it==index)
    {
        index++;
        ++it;
    }
    return VariableFactory::create();
}

void VariableFactorySet::add(const Variable& v) {
    predefined.insert(v);
}
