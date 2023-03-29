//
// Created by ramizouari on 12/03/23.
//

#ifndef MPGCPP_VARIABLE_H
#define MPGCPP_VARIABLE_H
#include <cstdint>
#include <compare>
#include <string>
#include <memory>
#include <vector>
#include <set>

class Variable {
    std::uint64_t id;
public:
    inline static constexpr std::uint16_t shift=32;
    Variable()= default;
    Variable(std::uint64_t id):id(id){}

    std::strong_ordering operator<=>(const Variable &other) const= default;
    std::strong_ordering operator<=>(const std::uint64_t &other) const;
    [[nodiscard]] virtual std::string name()
    {
        return std::to_string(id);
    }

    [[nodiscard]] std::uint64_t get_id() const
    {
        return id;
    }

    explicit operator std::uint64_t () const
    {
        return id;
    }
};

namespace std
{
    template<>
    struct hash<Variable>
    {
        std::size_t operator()(const Variable& v) const
        {
            return std::hash<std::uint64_t>()(v.get_id());
        }
    };
}

class VariableFactory
{
protected:
    std::uint64_t index=0;
    std::vector<std::unique_ptr<Variable>> variables;
public:
    VariableFactory()= default;
    virtual Variable* create();
    virtual ~VariableFactory()= default;
};

class VariableFactorySet : public VariableFactory
{
    std::set<Variable> predefined;
public:
    VariableFactorySet()= default;
    explicit VariableFactorySet(std::set<Variable> predefined);
    Variable* create() override;
    void add(const Variable& v);
    ~VariableFactorySet() override= default;
};

class VariableFactoryRange : public VariableFactory
{
public:
    explicit VariableFactoryRange(size_t n): VariableFactory()
    {
        index=n;
    }
};

namespace Print
{
    std::ostream &operator<<(std::ostream &os, const Variable &variable);
}

#endif //MPGCPP_VARIABLE_H
