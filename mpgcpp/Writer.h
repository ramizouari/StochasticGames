//
// Created by ramizouari on 15/03/23.
//

#ifndef MPGCPP_WRITER_H
#define MPGCPP_WRITER_H

#include <map>
#include <string>
#include <fstream>
#include <vector>

namespace Result {

    class Writer {
    public:
        virtual void writeHeader();
        virtual void writeFooter();
        virtual void writeData(const std::map<std::string, std::string> &data) = 0;
        virtual ~Writer() = default;
    };

    class StreamWriter : public Writer {
    protected:
        std::ostream *stream;
    public:
        StreamWriter(std::ostream *stream);
        void writeData(const std::map<std::string, std::string> &data) override;
        virtual ~StreamWriter() = default;
    };

    class FileWriter : public StreamWriter {
    protected:
        std::ofstream file;
    public:
        explicit FileWriter(const std::string& filename);
        virtual ~FileWriter() = default;
    };

    class CSVWriter : public FileWriter
    {
        char separator;
        const std::vector<std::string> header;
    public:
        explicit CSVWriter(const std::string& filename, char separator=',');
        explicit CSVWriter(const std::string& filename, const std::vector<std::string> &header, char separator=',');
        void writeHeader() override;
        void writeData(const std::map<std::string, std::string> &data) override;
    };

    class JSONWriter : public FileWriter
    {
        bool firstItem=true;
    public:
        explicit JSONWriter(const std::string& filename);
        void writeData(const std::map<std::string, std::string> &data) override;
        void writeHeader() override;
        void writeFooter() override;

    };

} // Result

#endif //MPGCPP_WRITER_H
