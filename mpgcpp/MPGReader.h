//
// Created by ramizouari on 13/03/23.
//

#ifndef MPGCPP_MPGREADER_H
#define MPGCPP_MPGREADER_H

#include <istream>
#include <fstream>
#include <boost/iostreams/filtering_streambuf.hpp>
#include <boost/iostreams/categories.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <boost/iostreams/filter/zlib.hpp>
#include <boost/iostreams/filter/bzip2.hpp>
#include <boost/iostreams/filter/lzma.hpp>
#include <regex>
#include <utility>
#include <filesystem>
#include "MeanPayoffGame.h"

template<MPG Game>
class MPGReader
{
public:
    virtual Game read()=0;
};

template<MPG Game>
class MPGStreamReader : public MPGReader<Game>
{
protected:
    std::istream &stream;
public:
    explicit MPGStreamReader(std::istream &stream) : stream(stream) {}
};

template<MPG Game>
class MPGWeightedEdgeReader : public MPGStreamReader<Game>
{
public:
    using MPGStreamReader<Game>::MPGStreamReader;
    using MPGStreamReader<Game>::stream;
    using weights_type=typename Game::weights_type;
    explicit MPGWeightedEdgeReader(std::istream &stream) : MPGStreamReader<Game>(stream) {}
    Game read() override
    {
        std::vector<std::tuple<std::uint64_t,std::uint64_t,weights_type>> edges;
        size_t nodes=0;
        while(!stream.eof())
        {
            std::uint64_t i,j;
            weights_type w;
            stream>>i>>j>>w;
            edges.emplace_back(i,j,w);
            nodes=std::max(nodes,std::max(i,j)+1);
        }
        Game game(nodes);
        for(auto [i,j,w]:edges)
            game.add_edge(i,j,w);
        return game;
    }
};

template<typename DecompressionStream>
class GenericDecompressor
{
protected:
    std::unique_ptr<std::istream> uncompressed;
    boost::iostreams::filtering_streambuf<boost::iostreams::input> in;
public:
    explicit GenericDecompressor(std::istream &compressed)
    {
        in.push(DecompressionStream());
        in.push(compressed);
        uncompressed=std::make_unique<std::istream>(&in);
        uncompressed->exceptions(std::ios::badbit);
    }
    std::istream& stream()
    {
        return *uncompressed;
    }
};

using GZDecompressor=GenericDecompressor<boost::iostreams::gzip_decompressor>;
using ZlibDecompressor=GenericDecompressor<boost::iostreams::zlib_decompressor>;
using BZ2Decompressor=GenericDecompressor<boost::iostreams::bzip2_decompressor>;
using LZMADecompressor=GenericDecompressor<boost::iostreams::lzma_decompressor>;

template<MPG Game,class DecompressionStream>
class MPGWeightedEdgeReaderCompressed : protected DecompressionStream, public MPGWeightedEdgeReader<Game>
{

public:
    using MPGWeightedEdgeReader<Game>::MPGWeightedEdgeReader;
    using MPGWeightedEdgeReader<Game>::stream;
    using MPGWeightedEdgeReader<Game>::read;
    explicit MPGWeightedEdgeReaderCompressed(std::istream &compressed) : DecompressionStream(compressed), MPGWeightedEdgeReader<Game>(*this->uncompressed)
    {
    }

};

template<MPG Game>
using MPGWeightedEdgeReaderGZ=MPGWeightedEdgeReaderCompressed<Game,GZDecompressor>;
template<typename Game>
using MPGWeightedEdgeReaderZlib=MPGWeightedEdgeReaderCompressed<Game,ZlibDecompressor>;
template<typename Game>
using MPGWeightedEdgeReaderBZ2=MPGWeightedEdgeReaderCompressed<Game,BZ2Decompressor>;
template<typename Game>
using MPGWeightedEdgeReaderLZMA=MPGWeightedEdgeReaderCompressed<Game,LZMADecompressor>;

template<MPG Game>
class MPGFileReader : public MPGReader<Game> {
    std::string file_name;
public:
  explicit MPGFileReader(std::string file_name): file_name(std::move(file_name)){}

    Game read() override
    {
      std::regex filename(R"(^.+(\.edgelist)(\.gz|\.bz2|\.zip|\.7z|)$)");
      std::smatch match;
      if (std::regex_search(file_name, match, filename))
      {
        if(match[1]!=".edgelist")
              throw std::runtime_error("Unknown file format");
        std::ifstream file(file_name, match[3]==""? std::ios::in: std::ios::in|std::ios::binary);
        if (match[2] == ".gz")
          return MPGWeightedEdgeReaderGZ<Game>(file).read();
        else if (match[2] == ".bz2")
          return MPGWeightedEdgeReaderBZ2<Game>(file).read();
        else if (match[2] == ".zip")
          return MPGWeightedEdgeReaderZlib<Game>(file).read();
        else if (match[2] == ".7z")
          return MPGWeightedEdgeReaderLZMA<Game>(file).read();
        else if (match[2] == "")
          return MPGWeightedEdgeReader<Game>(file).read();
        else
          throw std::runtime_error("Unknown file extension");
      }
      else
        throw std::runtime_error("Unknown file format");
    }
};

template<MPG Game>
Game mpg_from_file(const std::string& file_name)
{
    return MPGFileReader<Game>(file_name).read();
}

template<MPG Game>
Game mpg_from_file(const std::filesystem::path &file_name)
{
    return mpg_from_file<Game>(file_name.string());
}

#endif //MPGCPP_MPGREADER_H
