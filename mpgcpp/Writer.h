//
// Created by ramizouari on 15/03/23.
//

#ifndef MPGCPP_WRITER_H
#define MPGCPP_WRITER_H

#include <map>
#include <string>
#include <fstream>
#include <vector>
#include <memory>

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
        enum Type
        {
            STRING,
            NUMBER,
            BOOLEAN,
            ARRAY,
            OBJECT
        };
        explicit JSONWriter(const std::string& filename);
        void addType(const std::string &name, Type type);
        void writeData(const std::map<std::string, std::string> &data) override;
        void writeHeader() override;
        void writeFooter() override;
    private:
        std::map<std::string, Type> types;
    };

    class HumanWriter : public StreamWriter
    {
    public:
        explicit HumanWriter(std::ostream *stream);
        void writeHeader() override;
        void writeData(const std::map<std::string, std::string> &data) override;
    };

    class MultipleWriter : public Writer
    {
        std::vector<Writer*> writers;
    public:
        explicit MultipleWriter(const std::vector<Writer*> &writers={});
        void writeData(const std::map<std::string, std::string> &data) override;
        void writeHeader() override;
        void writeFooter() override;
        void addWriter(Writer *writer);
    };

    class MultipleWriterUnique : public Writer
    {
        std::vector<std::unique_ptr<Writer>> writers;
    public:
        explicit MultipleWriterUnique();
        void writeData(const std::map<std::string, std::string> &data) override;
        void writeHeader() override;
        void writeFooter() override;
        void addWriter(Writer *writer);
    };

    class ParallelWriter : public Writer {
        std::mutex mutex;
        Writer & writer;
    public:
        explicit ParallelWriter(Writer &writer);
        void writeData(const std::map<std::string, std::string> &data) override;
        void writeHeader() override;
        void writeFooter() override;
    };
} // Result

#endif //MPGCPP_WRITER_H
