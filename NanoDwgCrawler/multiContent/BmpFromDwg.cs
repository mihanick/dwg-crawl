using DwgDump.Enitites;
using Multicad;
using Multicad.AplicationServices;
using Multicad.DatabaseServices;
using Multicad.DatabaseServices.StandardObjects;
using Multicad.Dimensions;
using Multicad.Geometry;
using Multicad.Symbols;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DwgDump.MultiContent
{
	public static class BmpFromDwg
	{
		/// <summary>
		/// Создает превью из формата размером <paramref name="imageSize"/> пикселей в формате png
		/// </summary>
		/// <param name="format"></param>
		/// <returns></returns>
		public static (Bitmap, BoundBlock) CreatePreview(List<McObjectId> selectedIds, int imageSize = 512, BoundBlock bound = null)
		{
			var bbOx = new BoundBlock();
			if (bound == null)
			{
				foreach (var entId in selectedIds)
				{
					McDbEntity dbe = (entId.GetObject())?.Cast<McDbEntity>();
					if (dbe != null)
						bbOx = bbOx.MergeWith(dbe.BoundingBox);
				}
			}
			else
				bbOx = bound;

			McEmfParams p = new McEmfParams()
			{
				UseColor = true,
				UseWidth = true,
				BgrColor = System.Drawing.Color.White,
				PntOfLeftBtm = bbOx.BasePoint,
				SizeX = bbOx.SizeByX,
				SizeY = bbOx.SizeByY,
				Frame = 3,
				SquareEmf = false,
				Quite = true,
				Scale = 10
			};

			Metafile mf = McNativeGate.CreateEmf(selectedIds, p);

			// BUG: mf could be null, it will throw, but nanocad should handle it

			int bmpHeight, bmpWidth;
			if (mf.Height > mf.Width)
			{
				bmpHeight = imageSize;
				bmpWidth = (int)(mf.Width * imageSize / mf.Height);
			}
			else
			{
				bmpWidth = imageSize;
				bmpHeight = (int)(mf.Height * imageSize / mf.Width);
			}

			Bitmap d = new Bitmap(mf, bmpWidth, bmpHeight);

			using (Bitmap b = new Bitmap(d))
			{
				// string path = Path.Combine(Path.GetTempFileName() + ".png");
				//b.Save(path, ImageFormat.Png);
			}

			return (d, bbOx);
		}

	}
}
