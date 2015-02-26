using System;
using System.Runtime.Serialization;
using System.IO;

namespace Crawl
{
    [DataContract]
    public class CrawlDocument
    {
        [DataMember]
        public string Path;
        [DataMember]
        public string FileId;
        [DataMember]
        public string Hash;

        public string docJson;

        public CrawlDocument(string dwgPath="")
        {
                this.Path = dwgPath;
                if(File.Exists(dwgPath))
                    this.Hash = Crawl.UtilityHash.HashFile(dwgPath);
         
                this.FileId = Guid.NewGuid().ToString();
                this.docJson = jsonHelper.To<CrawlDocument>(this);
        }
    }
}
