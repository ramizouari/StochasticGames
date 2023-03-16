//
// Created by ramizouari on 15/03/23.
//

#include <ostream>
#include "Writer.h"

namespace Result {
    void Writer::writeHeader() {

    }

    void Writer::writeFooter() {

    }

    StreamWriter::StreamWriter(std::ostream *stream) : stream(stream)
    {
    }

    void StreamWriter::writeData(const std::map<std::string, std::string> &data) {
        for (auto &item : data)
            (*stream) << item.first << " = " << item.second << std::endl;
    }

    FileWriter::FileWriter(const std::string& filename): file(filename), StreamWriter(&file) {

    }


    CSVWriter::CSVWriter(const std::string &filename, char separator) : FileWriter(filename), separator(separator) {

    }

    CSVWriter::CSVWriter(const std::string &filename, const std::vector<std::string> &header, char separator) : FileWriter(filename), separator(separator), header(header) {

    }

    void CSVWriter::writeHeader()
    {
        for(int i=0;i<header.size();i++)
        {
            file<<header[i];
            if(i<header.size()-1)
                file<<separator;
            else file<<std::endl;
        }
    }

    void CSVWriter::writeData(const std::map<std::string, std::string> &data) {
        for(int i=0;i<header.size();i++)
        {
            if(data.contains(header[i]))
                file << '"' << data.at(header[i]) << '"';
            if(i<header.size()-1)
                file<<separator;
            else file<<std::endl;
        }
    }

    JSONWriter::JSONWriter(const std::string &filename) : FileWriter(filename) {
    }

    void JSONWriter::writeData(const std::map<std::string, std::string> &data) {
        if(firstItem)
            firstItem=false;
        else
            file<<",\n";
        file<<"{";
        for(auto &item:data)
        {
            switch (types[item.first])
            {
                case STRING:
                    file<<"\""<<item.first<<"\""<<":"<<"\""<<item.second<<"\"";
                    break;
                default:
                    file<<"\""<<item.first<<"\""<<":"<<item.second;
                    break;
            }
            if(item.first!=data.rbegin()->first)
                file<<",\n";
        }
        file<<"}"<<std::endl;
    }

    void JSONWriter::writeHeader() {
        file<<"[\n";
    }
    void JSONWriter::writeFooter() {
        file<<"]";
    }

    void JSONWriter::addType(const std::string &name, JSONWriter::Type type) {
        types[name]=type;
    }

    MultipleWriter::MultipleWriter(const std::vector<Writer *> &writers): writers(writers)
    {
    }

    void MultipleWriter::writeData(const std::map<std::string, std::string> &data)
    {
        for(auto writer:writers)
            writer->writeData(data);
    }

    void MultipleWriter::writeHeader() {
        for(auto writer:writers)
            writer->writeHeader();
    }

    void MultipleWriter::writeFooter() {
        for(auto writer:writers)
            writer->writeFooter();
    }

    void MultipleWriter::addWriter(Writer *writer)
    {
        writers.push_back(writer);
    }

    MultipleWriterUnique::MultipleWriterUnique() = default;

    void MultipleWriterUnique::writeHeader() {
        for(auto& writer:writers)
            writer->writeHeader();
    }

    void MultipleWriterUnique::writeFooter() {
        for(auto& writer:writers)
            writer->writeFooter();
    }

    void MultipleWriterUnique::writeData(const std::map<std::string, std::string> &data) {
        for(auto& writer:writers)
            writer->writeData(data);
    }

    void MultipleWriterUnique::addWriter(Writer *writer) {
        writers.emplace_back(writer);
    }

    HumanWriter::HumanWriter(std::ostream *stream) : StreamWriter(stream) {

    }

    void HumanWriter::writeHeader() {
        (*stream)<<"Result:"<<std::endl;
    }

    void HumanWriter::writeData(const std::map<std::string, std::string> &data) {
        (*stream) << "Results:" << std::endl;
        for(auto &item:data)
            (*stream)<< '\t' <<item.first<<": "<<item.second<< '\n';
    }

    void ParallelWriter::writeHeader() {
        writer.writeHeader();
    }

    void ParallelWriter::writeFooter() {
        writer.writeFooter();
    }

    void ParallelWriter::writeData(const std::map<std::string, std::string> &data) {
        mutex.lock();
        writer.writeData(data);
        mutex.unlock();
    }

    ParallelWriter::ParallelWriter(Writer &writer) : writer(writer)
    {
    }
} // Result