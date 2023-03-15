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
            file<<data.at(header[i]);
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
            file<<"\""<<item.first<<"\""<<":"<<"\""<<item.second<<"\"";
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
} // Result